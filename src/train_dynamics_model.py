import json
import os

import joblib
import numpy as np
import torch

torch.set_num_threads(1)
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ACTION_NAMES, DATA_PATH, ENV_COLUMNS, MODEL_DIR, NUM_ACTIONS, SHEET_NAME
from data_utils import chronological_split, make_dynamics_dataset
from models import DynamicsMLP


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(X, y, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_mae(model, X, y, y_scaler):
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_s)
    return mean_absolute_error(y, pred, multioutput="raw_values"), pred


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    _, X, y, lag_features = make_dynamics_dataset(DATA_PATH, SHEET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    loader = make_loader(X_train_s, y_train_s)
    model = DynamicsMLP(input_dim=X_train_s.shape[1], output_dim=y_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    waited = 0
    patience = 30

    for epoch in range(1, 301):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE)),
                torch.tensor(y_val_s, dtype=torch.float32).to(DEVICE),
            ).item()

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1

        if waited >= patience:
            break

    model.load_state_dict(best_state)
    test_mae, _ = evaluate_mae(model, X_test_s, y_test, y_scaler)

    print("\nTest MAE by next-hour environment-variable delta")
    for name, value in zip(ENV_COLUMNS, test_mae):
        print(f"{name}: {value:.4f}")

    torch.save(model.state_dict(), f"{MODEL_DIR}/dynamics_mlp.pt")
    joblib.dump(x_scaler, f"{MODEL_DIR}/dynamics_x_scaler.joblib")
    joblib.dump(y_scaler, f"{MODEL_DIR}/dynamics_y_scaler.joblib")
    with open(f"{MODEL_DIR}/dynamics_metadata.json", "w") as f:
        json.dump({
            "model_type": "ann_lag_dynamics",
            "features": lag_features,
            "action_features": [f"action_{ACTION_NAMES[i]}" for i in range(NUM_ACTIONS)],
            "targets": ENV_COLUMNS,
            "num_actions": NUM_ACTIONS,
            "input_dim": int(X_train_s.shape[1]),
            "output_dim": int(y_train.shape[1]),
        }, f, indent=2)
    print("\nSaved dynamics model to models/dynamics_mlp.pt")


if __name__ == "__main__":
    main()
