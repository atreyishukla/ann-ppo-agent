import json
import sys

import joblib
import torch

from config import ACTION_BITS, ACTION_NAMES, MODEL_DIR, NUM_ACTIONS
from data_utils import build_current_feature_row, build_lag_feature_row, build_sequence_array
from models import GreenhouseLSTM, GreenhouseMLP


def action_payload(action_id):
    bits = ACTION_BITS[action_id]
    fan_on = int(bits["fan_on"])
    heater_on = int(bits["heater_on"])
    return {
        "action_id": action_id,
        "action": ACTION_NAMES[action_id],
        "fan_on": fan_on,
        "fan_off": 1 - fan_on,
        "heater_on": heater_on,
        "heater_off": 1 - heater_on,
        "both_off": int(fan_on == 0 and heater_on == 0),
        "both_on": int(fan_on == 1 and heater_on == 1),
    }


def load_model():
    with open(f"{MODEL_DIR}/supervised_best_metadata.json") as f:
        metadata = json.load(f)
    scaler = joblib.load(f"{MODEL_DIR}/supervised_best_scaler.joblib")
    output_dim = int(metadata.get("output_dim", NUM_ACTIONS))

    if metadata["model_type"] == "lstm_24h":
        model = GreenhouseLSTM(
            input_dim=metadata["input_dim"],
            output_dim=output_dim,
            hidden_dim=metadata.get("hidden_dim", 32),
            num_layers=metadata.get("num_layers", 1),
        )
    else:
        model = GreenhouseMLP(input_dim=metadata["input_dim"], output_dim=output_dim)

    model.load_state_dict(torch.load(f"{MODEL_DIR}/supervised_best.pt", map_location="cpu"))
    model.eval()
    return model, scaler, metadata


def build_model_input(payload, scaler, metadata):
    model_type = metadata["model_type"]
    features = metadata["features"]

    if model_type == "ann_current":
        X = build_current_feature_row(payload)
        return torch.tensor(scaler.transform(X), dtype=torch.float32)

    if model_type == "ann_lag":
        X = build_lag_feature_row(payload, features)
        return torch.tensor(scaler.transform(X), dtype=torch.float32)

    if model_type == "lstm_24h":
        window = metadata["sequence_window"]
        X = build_sequence_array(payload, features, window)
        shape = X.shape
        X_scaled = scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)
        return torch.tensor(X_scaled, dtype=torch.float32)

    raise ValueError(f"Unknown model type: {model_type}")


def predict(payload):
    model, scaler, metadata = load_model()
    X_tensor = build_model_input(payload, scaler, metadata)

    with torch.no_grad():
        probs = torch.softmax(model(X_tensor), dim=1).numpy()[0]

    action_id = int(probs.argmax())
    result = action_payload(action_id)
    result.update({
        "model_type": metadata["model_type"],
        "confidence": float(probs[action_id]),
        "probabilities": {ACTION_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        "needs_history": bool(metadata.get("needs_history", False)),
    })
    return result


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "Date_time": "2026-01-01 08:00:00",
        "T_outside": -8.0,
        "T_inside": 16.0,
        "T_floor_inside": 20.0,
        "T_floor_outside": -6.0,
        "SR_direct_outside": 100.0,
    }
    print(json.dumps(predict(payload), indent=2))
