from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from config import ACTION_NAMES, DATA_PATH, MODEL_DIR, NUM_ACTIONS, SHEET_NAME
from data_utils import make_lag_dataset, stratified_three_way_split
from models import GreenhouseMLP


ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "reports" / "ann_lag_graphs"
DATA_DIR = ROOT / "reports" / "ann_lag_data"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    path = GRAPH_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def load_ann_lag_model():
    metadata_path = ROOT / MODEL_DIR / "supervised_ann_lag_metadata.json"
    model_path = ROOT / MODEL_DIR / "supervised_ann_lag.pt"
    scaler_path = ROOT / MODEL_DIR / "supervised_ann_lag_scaler.joblib"

    if not metadata_path.exists():
        metadata_path = ROOT / MODEL_DIR / "supervised_best_metadata.json"
        model_path = ROOT / MODEL_DIR / "supervised_best.pt"
        scaler_path = ROOT / MODEL_DIR / "supervised_best_scaler.joblib"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    scaler = joblib.load(scaler_path)

    model = GreenhouseMLP(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"]
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler, metadata


def get_ann_lag_predictions():
    df, X, y, features = make_lag_dataset(DATA_PATH, SHEET_NAME)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y)

    model, scaler, metadata = load_ann_lag_model()

    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(X_test_scaled, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()
        y_pred = probabilities.argmax(axis=1)

    y_test_np = np.asarray(y_test)

    results = pd.DataFrame({
        "actual_action_id": y_test_np,
        "predicted_action_id": y_pred,
        "actual_action": [ACTION_NAMES[int(i)] for i in y_test_np],
        "predicted_action": [ACTION_NAMES[int(i)] for i in y_pred],
        "confidence": probabilities.max(axis=1),
    })

    results.to_csv(DATA_DIR / "ann_lag_predictions.csv", index=False)

    report = classification_report(
        y_test_np,
        y_pred,
        labels=list(range(NUM_ACTIONS)),
        target_names=[ACTION_NAMES[i] for i in range(NUM_ACTIONS)],
        output_dict=True,
        zero_division=0,
    )

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(DATA_DIR / "ann_lag_classification_report.csv")

    return y_test_np, y_pred, results, metrics_df


def plot_historical_action_distribution():
    df, X, y, features = make_lag_dataset(DATA_PATH, SHEET_NAME)

    counts = y.value_counts().sort_index()
    labels = [ACTION_NAMES[i] for i in counts.index]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts.values)
    plt.title("Historical Action Distribution")
    plt.xlabel("Action state")
    plt.ylabel("Number of records")
    plt.xticks(rotation=20)
    save_plot("01_historical_action_distribution.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_ACTIONS)))
    labels = [ACTION_NAMES[i] for i in range(NUM_ACTIONS)]

    plt.figure(figsize=(7, 5))
    plt.imshow(cm)
    plt.title("ANN-lag Confusion Matrix")
    plt.xlabel("Predicted action")
    plt.ylabel("Actual action")
    plt.xticks(range(NUM_ACTIONS), labels, rotation=25, ha="right")
    plt.yticks(range(NUM_ACTIONS), labels)

    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    save_plot("02_ann_lag_confusion_matrix.png")


def plot_per_class_f1(metrics_df):
    class_rows = [ACTION_NAMES[i] for i in range(NUM_ACTIONS)]
    f1_scores = metrics_df.loc[class_rows, "f1-score"]

    plt.figure(figsize=(8, 4))
    plt.bar(class_rows, f1_scores.values)
    plt.title("ANN-lag F1 Score by Action")
    plt.xlabel("Action state")
    plt.ylabel("F1 score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    save_plot("03_ann_lag_f1_by_action.png")


def plot_actual_vs_predicted(results):
    actual_counts = results["actual_action"].value_counts()
    predicted_counts = results["predicted_action"].value_counts()

    labels = [ACTION_NAMES[i] for i in range(NUM_ACTIONS)]
    actual = [actual_counts.get(label, 0) for label in labels]
    predicted = [predicted_counts.get(label, 0) for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, actual, width, label="Actual")
    plt.bar(x + width / 2, predicted, width, label="Predicted")
    plt.title("ANN-lag Actual vs Predicted Action Counts")
    plt.xlabel("Action state")
    plt.ylabel("Number of test records")
    plt.xticks(x, labels, rotation=20)
    plt.legend()
    save_plot("04_ann_lag_actual_vs_predicted.png")


def main():
    y_true, y_pred, results, metrics_df = get_ann_lag_predictions()

    plot_historical_action_distribution()
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_f1(metrics_df)
    plot_actual_vs_predicted(results)

    print("\nSaved ANN-lag graph data to reports/ann_lag_data/")
    print("Saved ANN-lag graphs to reports/ann_lag_graphs/")


if __name__ == "__main__":
    main()