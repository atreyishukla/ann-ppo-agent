import json
import sys

from predict_dl import predict as predict_supervised


def predict(payload):
    mode = payload.get("controller", payload.get("mode", "supervised"))
    if mode == "rl":
        from predict_rl import predict as predict_rl
        return predict_rl(payload)
    return predict_supervised(payload)


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "controller": "supervised",
        "Date_time": "2026-01-01 08:00:00",
        "T_outside": -8.0,
        "T_inside": 16.0,
        "T_floor_inside": 20.0,
        "T_floor_outside": -6.0,
        "SR_direct_outside": 100.0,
        "previous_action": 0,
    }
    print(json.dumps(predict(payload), indent=2))
