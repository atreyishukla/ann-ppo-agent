import json
import os

import joblib
import torch

from config import DATA_PATH, MODEL_DIR, SHEET_NAME
from train_supervised_compare import train_ann_lag


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    model, scaler, metadata, metrics = train_ann_lag()
    metadata["forced_model"] = "ann_lag"
    metadata["note"] = "ANN lag model selected intentionally for deployment."

    torch.save(model.state_dict(), f"{MODEL_DIR}/supervised_best.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/supervised_best_scaler.joblib")

    with open(f"{MODEL_DIR}/supervised_best_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    torch.save(model.state_dict(), f"{MODEL_DIR}/supervised_ann_lag.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/supervised_ann_lag_scaler.joblib")

    with open(f"{MODEL_DIR}/supervised_ann_lag_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved ANN lag model for deployment")
    print("Files used by Node-RED:")
    print("models/supervised_best.pt")
    print("models/supervised_best_scaler.joblib")
    print("models/supervised_best_metadata.json")
    print("\nBackup ANN lag files:")
    print("models/supervised_ann_lag.pt")
    print("models/supervised_ann_lag_scaler.joblib")
    print("models/supervised_ann_lag_metadata.json")
    print("\nMetrics:")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
