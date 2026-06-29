import json
import os

import joblib
import numpy as np
import torch

torch.set_num_threads(1)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ACTION_NAMES, DATA_PATH, MODEL_DIR, NUM_ACTIONS, SHEET_NAME, SUPERVISED_FEATURES
from data_utils import make_lag_dataset, make_sequence_dataset, make_supervised_dataset, stratified_three_way_split
from models import GreenhouseLSTM, GreenhouseMLP


BATCH_SIZE = 256
EPOCHS = 80
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def class_weights(y):
    counts = np.bincount(np.asarray(y), minlength=NUM_ACTIONS)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def to_numpy(value):
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    X_tensor = torch.tensor(to_numpy(X), dtype=torch.float32)
    y_tensor = torch.tensor(to_numpy(y), dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def fit_scaler_for_2d(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return scaler, X_train_s.astype(np.float32), X_val_s.astype(np.float32), X_test_s.astype(np.float32)


def fit_scaler_for_3d(X_train, X_val, X_test):
    scaler = StandardScaler()
    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape
    X_train_s = scaler.fit_transform(X_train.reshape(-1, train_shape[-1])).reshape(train_shape)
    X_val_s = scaler.transform(X_val.reshape(-1, val_shape[-1])).reshape(val_shape)
    X_test_s = scaler.transform(X_test.reshape(-1, test_shape[-1])).reshape(test_shape)
    return scaler, X_train_s.astype(np.float32), X_val_s.astype(np.float32), X_test_s.astype(np.float32)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(to_numpy(X), dtype=torch.float32).to(DEVICE))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_np = to_numpy(y)
    return {
        "accuracy": accuracy_score(y_np, pred),
        "balanced_accuracy": balanced_accuracy_score(y_np, pred),
        "macro_f1": f1_score(y_np, pred, average="macro", zero_division=0),
        "predictions": pred,
    }


def train_one_model(model_name, model, X_train, X_val, X_test, y_train, y_val, y_test):
    model = model.to(DEVICE)
    train_loader = make_loader(X_train, y_train)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_state = None
    best_val_f1 = -1.0
    waited = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, X_val, y_val)
        if epoch == 1 or epoch % 25 == 0:
            print(f"{model_name} epoch={epoch} val_macro_f1={val_metrics['macro_f1']:.4f} val_accuracy={val_metrics['accuracy']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1

        if waited >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, X_test, y_test)
    y_test_np = to_numpy(y_test)

    print(f"\n{model_name} test metrics")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {test_metrics['macro_f1']:.4f}")
    print(classification_report(
        y_test_np,
        test_metrics["predictions"],
        labels=list(range(NUM_ACTIONS)),
        target_names=[ACTION_NAMES[i] for i in range(NUM_ACTIONS)],
        zero_division=0,
    ))
    print("Confusion matrix [fan_off_heater_off, fan_off_heater_on, fan_on_heater_off, fan_on_heater_on]:")
    print(confusion_matrix(y_test_np, test_metrics["predictions"], labels=list(range(NUM_ACTIONS))))

    return model, test_metrics


def train_ann_current():
    _, X, y = make_supervised_dataset(DATA_PATH, SHEET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y)
    scaler, X_train_s, X_val_s, X_test_s = fit_scaler_for_2d(X_train, X_val, X_test)
    model = GreenhouseMLP(input_dim=X_train_s.shape[1], output_dim=NUM_ACTIONS)
    trained, metrics = train_one_model("ann_current", model, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)
    metadata = {
        "model_type": "ann_current",
        "features": list(SUPERVISED_FEATURES),
        "input_dim": X_train_s.shape[1],
        "output_dim": NUM_ACTIONS,
        "actions": ACTION_NAMES,
        "needs_history": False,
    }
    return trained, scaler, metadata, metrics


def train_ann_lag():
    _, X, y, features = make_lag_dataset(DATA_PATH, SHEET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y)
    scaler, X_train_s, X_val_s, X_test_s = fit_scaler_for_2d(X_train, X_val, X_test)
    model = GreenhouseMLP(input_dim=X_train_s.shape[1], output_dim=NUM_ACTIONS)
    trained, metrics = train_one_model("ann_lag", model, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)
    metadata = {
        "model_type": "ann_lag",
        "features": features,
        "input_dim": X_train_s.shape[1],
        "output_dim": NUM_ACTIONS,
        "actions": ACTION_NAMES,
        "needs_history": True,
    }
    return trained, scaler, metadata, metrics


def train_lstm(window=24):
    _, X, y, features = make_sequence_dataset(DATA_PATH, SHEET_NAME, window=window)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y)
    scaler, X_train_s, X_val_s, X_test_s = fit_scaler_for_3d(X_train, X_val, X_test)
    model = GreenhouseLSTM(input_dim=X_train_s.shape[2], output_dim=NUM_ACTIONS, hidden_dim=32, num_layers=1)
    trained, metrics = train_one_model("lstm_24h", model, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)
    metadata = {
        "model_type": "lstm_24h",
        "features": features,
        "input_dim": X_train_s.shape[2],
        "output_dim": NUM_ACTIONS,
        "sequence_window": window,
        "hidden_dim": 32,
        "num_layers": 1,
        "actions": ACTION_NAMES,
        "needs_history": True,
    }
    return trained, scaler, metadata, metrics


def save_best_model(model, scaler, metadata):
    torch.save(model.state_dict(), f"{MODEL_DIR}/supervised_best.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/supervised_best_scaler.joblib")
    with open(f"{MODEL_DIR}/supervised_best_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    torch.save(model.state_dict(), f"{MODEL_DIR}/supervised_mlp.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/supervised_scaler.joblib")
    with open(f"{MODEL_DIR}/supervised_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    _, _, y = make_supervised_dataset(DATA_PATH, SHEET_NAME)
    print("Class counts:")
    print(y.value_counts().sort_index())
    print(f"Device: {DEVICE}\n")

    candidates = []
    for trainer in [train_ann_current, train_ann_lag, train_lstm]:
        model, scaler, metadata, metrics = trainer()
        candidates.append({
            "model": model,
            "scaler": scaler,
            "metadata": metadata,
            "metrics": metrics,
        })

    print("\nSummary")
    for item in sorted(candidates, key=lambda x: x["metrics"]["macro_f1"], reverse=True):
        name = item["metadata"]["model_type"]
        print(
            f"{name}: accuracy={item['metrics']['accuracy']:.4f}, "
            f"balanced_accuracy={item['metrics']['balanced_accuracy']:.4f}, "
            f"macro_f1={item['metrics']['macro_f1']:.4f}"
        )

    best = max(candidates, key=lambda x: x["metrics"]["macro_f1"])
    save_best_model(best["model"], best["scaler"], best["metadata"])
    print(f"\nSaved best supervised model: {best['metadata']['model_type']}")
    print("Files: models/supervised_best.pt, models/supervised_best_scaler.joblib, models/supervised_best_metadata.json")


if __name__ == "__main__":
    main()
