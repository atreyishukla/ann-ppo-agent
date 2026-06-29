import json
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import ACTION_BITS, ACTION_NAMES, DEFAULT_COMFORT_BAND, DEFAULT_TARGET_TEMP, ENV_COLUMNS, MODEL_DIR
from data_utils import build_time_features


def action_payload(action_id):
    action_id = int(action_id)
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


def _metadata_defaults():
    metadata_path = f"{MODEL_DIR}/rl_observation_metadata.json"
    if not os.path.exists(metadata_path):
        return DEFAULT_TARGET_TEMP, DEFAULT_COMFORT_BAND
    with open(metadata_path) as f:
        metadata = json.load(f)
    return float(metadata.get("target_temp", DEFAULT_TARGET_TEMP)), float(metadata.get("comfort_band", DEFAULT_COMFORT_BAND))


def build_raw_observation(payload):
    trained_target, trained_band = _metadata_defaults()
    target = float(payload.get("target_temp", trained_target))
    band = float(payload.get("comfort_band", trained_band))

    time_values = build_time_features(pd.to_datetime(payload["Date_time"]))
    previous_action = float(payload.get("previous_action", 0))
    env_values = [float(payload[col]) for col in ENV_COLUMNS]
    t_inside = env_values[1]
    temp_error = t_inside - target
    below_band = max((target - band) - t_inside, 0.0)
    above_band = max(t_inside - (target + band), 0.0)

    obs = []
    obs.extend(env_values)
    obs.extend([
        float(time_values["hour_sin"]),
        float(time_values["hour_cos"]),
        float(time_values["month_sin"]),
        float(time_values["month_cos"]),
        previous_action,
        temp_error,
        below_band,
        above_band,
    ])
    return np.asarray(obs, dtype=np.float32)


def normalize_observation(raw_obs):
    metadata_path = f"{MODEL_DIR}/rl_observation_metadata.json"
    if not os.path.exists(metadata_path):
        return raw_obs

    with open(metadata_path) as f:
        metadata = json.load(f)
    mean = np.asarray(metadata["obs_mean"], dtype=np.float32)
    std = np.asarray(metadata["obs_std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((raw_obs - mean) / std).astype(np.float32)


def simple_direction_hint(payload):
    trained_target, trained_band = _metadata_defaults()
    target = float(payload.get("target_temp", trained_target))
    band = float(payload.get("comfort_band", trained_band))
    t_inside = float(payload["T_inside"])
    if t_inside < target - band:
        return "too_cold_expected_heater_on"
    if t_inside > target + band:
        return "too_hot_expected_fan_on"
    return "comfortable_expected_both_off"


def predict(payload):
    from stable_baselines3 import PPO

    model = PPO.load(f"{MODEL_DIR}/ppo_greenhouse_agent")
    raw_obs = build_raw_observation(payload)
    obs = normalize_observation(raw_obs)
    action, _ = model.predict(obs, deterministic=True)
    action_id = int(action)

    result = action_payload(action_id)
    result["model_type"] = "ppo_rl_agent_4_state_ann_lag_dynamics_no_final_override"
    result["target_temp"] = float(payload.get("target_temp", _metadata_defaults()[0]))
    result["comfort_band"] = float(payload.get("comfort_band", _metadata_defaults()[1]))
    result["direction_hint_not_override"] = simple_direction_hint(payload)
    result["final_override_used"] = 0
    return result


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "Date_time": "2026-01-01 08:00:00",
        "T_outside": -8.0,
        "T_inside": 16.0,
        "T_floor_inside": 20.0,
        "T_floor_outside": -6.0,
        "SR_direct_outside": 100.0,
        "previous_action": 0,
    }
    print(json.dumps(predict(payload), indent=2))
