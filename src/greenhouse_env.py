import json

import gymnasium as gym
import joblib
import numpy as np
import torch
from gymnasium import spaces

from config import (
    ACTION_BITS,
    ACTION_NAMES,
    BOTH_ON_PENALTY,
    COMFORT_ACTIVE_PENALTY,
    COMFORT_BOTH_OFF_BONUS,
    DATA_PATH,
    DEFAULT_COMFORT_BAND,
    DEFAULT_TARGET_TEMP,
    ENV_COLUMNS,
    FAN_ENERGY_PENALTY,
    FAN_FLOOR_DROP,
    FAN_TEMP_DROP,
    HEATER_ENERGY_PENALTY,
    HEATER_FLOOR_GAIN,
    HEATER_TEMP_GAIN,
    MODEL_DIR,
    NUM_ACTIONS,
    RIGHT_ACTION_BONUS,
    SHEET_NAME,
    SUPERVISED_FEATURES,
    SWITCHING_PENALTY,
    WRONG_ACTION_PENALTY,
)
from data_utils import (
    action_to_one_hot,
    add_action_labels,
    add_time_features,
    build_lag_features_from_rows,
    load_sheet,
)
from models import DynamicsMLP


class GreenhouseEnv(gym.Env):
    """4-state greenhouse RL environment.

    The ANN-lag dynamics model predicts the natural next-hour greenhouse trend.
    Physics-informed action effects are applied inside the simulator so RL learns
    a causal policy instead of memorizing sparse historical controller behavior.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path=DATA_PATH,
        sheet_name=SHEET_NAME,
        target_temp=DEFAULT_TARGET_TEMP,
        comfort_band=DEFAULT_COMFORT_BAND,
        episode_len=24,
        curriculum_prob=0.55,
    ):
        super().__init__()
        self.df = load_sheet(data_path, sheet_name)
        self.df = add_action_labels(add_time_features(self.df)).reset_index(drop=True)
        self.df = self.df.dropna(subset=SUPERVISED_FEATURES + ["action"]).reset_index(drop=True)

        with open(f"{MODEL_DIR}/dynamics_metadata.json") as f:
            self.dynamics_metadata = json.load(f)

        self.lag_features = self.dynamics_metadata["features"]
        self.target_temp = float(target_temp)
        self.comfort_band = float(comfort_band)
        self.episode_len = int(episode_len)
        self.curriculum_prob = float(curriculum_prob)
        self.max_lag = 24

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.obs_mean, self.obs_std = self._build_observation_stats()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(13,), dtype=np.float32)

        self.x_scaler = joblib.load(f"{MODEL_DIR}/dynamics_x_scaler.joblib")
        self.y_scaler = joblib.load(f"{MODEL_DIR}/dynamics_y_scaler.joblib")
        self.dynamics = DynamicsMLP(
            input_dim=int(self.dynamics_metadata["input_dim"]),
            output_dim=len(ENV_COLUMNS),
        )
        self.dynamics.load_state_dict(torch.load(f"{MODEL_DIR}/dynamics_mlp.pt", map_location="cpu"))
        self.dynamics.eval()

    def _build_observation_stats(self):
        obs = []
        for _, row in self.df.iterrows():
            env_values = row[ENV_COLUMNS].to_numpy(dtype=np.float32)
            time_values = row[["hour_sin", "hour_cos", "month_sin", "month_cos"]].to_numpy(dtype=np.float32)
            obs.append(self._compose_raw_observation(env_values, time_values, previous_action=0))
        obs = np.asarray(obs, dtype=np.float32)
        mean = obs.mean(axis=0).astype(np.float32)
        std = obs.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    def save_observation_stats(self, path=f"{MODEL_DIR}/rl_observation_metadata.json"):
        with open(path, "w") as f:
            json.dump({
                "obs_mean": self.obs_mean.tolist(),
                "obs_std": self.obs_std.tolist(),
                "target_temp": self.target_temp,
                "comfort_band": self.comfort_band,
                "observation_order": (
                    ENV_COLUMNS
                    + ["hour_sin", "hour_cos", "month_sin", "month_cos", "previous_action"]
                    + ["temp_error", "below_comfort_band", "above_comfort_band"]
                ),
                "note": "RL model uses normalized observations with explicit target-temperature error features.",
            }, f, indent=2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        min_start = self.max_lag
        max_start = len(self.df) - self.episode_len - 2
        self.idx = int(self.np_random.integers(min_start, max_start))
        self.step_count = 0
        self.previous_action = 0
        self.env_values = self.df.loc[self.idx, ENV_COLUMNS].to_numpy(dtype=np.float32)

        self.history_rows = []
        for i in range(self.idx - self.max_lag, self.idx):
            self.history_rows.append(self._row_from_df(i))

        if self.np_random.random() < self.curriculum_prob:
            self._apply_curriculum_start_state()

        return self._normalized_observation(), {}

    def _apply_curriculum_start_state(self):
        r = float(self.np_random.random())

        if r < 0.22:
            scenario = 0
        elif r < 0.40:
            scenario = 1
        elif r < 0.62:
            scenario = 2
        elif r < 0.80:
            scenario = 3
        else:
            scenario = 4

        if scenario == 0:
            # Severe cold case: heater should clearly be best.
            outside = float(self.np_random.uniform(-28, -8))
            inside = float(self.np_random.uniform(self.target_temp - 15, self.target_temp - 6))
            floor_inside = inside + float(self.np_random.uniform(-4, 2))
            floor_outside = outside + float(self.np_random.uniform(-2, 4))
            solar = float(self.np_random.uniform(0, 120))

        elif scenario == 1:
            # Mild cold case: heater should usually be best.
            outside = float(self.np_random.uniform(-10, 8))
            inside = float(self.np_random.uniform(self.target_temp - 6, self.target_temp - 1.5))
            floor_inside = inside + float(self.np_random.uniform(-3, 3))
            floor_outside = outside + float(self.np_random.uniform(-2, 4))
            solar = float(self.np_random.uniform(0, 350))

        elif scenario == 2:
            # Severe hot case: fan should clearly be best.
            outside = float(self.np_random.uniform(27, 40))
            inside = float(self.np_random.uniform(self.target_temp + 6, self.target_temp + 16))
            floor_inside = inside + float(self.np_random.uniform(-2, 5))
            floor_outside = outside + float(self.np_random.uniform(-2, 3))
            solar = float(self.np_random.uniform(500, 950))

        elif scenario == 3:
            # Mild hot case: fan should usually be best.
            outside = float(self.np_random.uniform(18, 34))
            inside = float(self.np_random.uniform(self.target_temp + 1.5, self.target_temp + 7))
            floor_inside = inside + float(self.np_random.uniform(-2, 4))
            floor_outside = outside + float(self.np_random.uniform(-2, 3))
            solar = float(self.np_random.uniform(250, 850))

        else:
            # Comfortable case: both off should clearly be best.
            outside = float(self.np_random.uniform(12, 26))
            inside = float(self.np_random.uniform(self.target_temp - 0.7, self.target_temp + 0.7))
            floor_inside = inside + float(self.np_random.uniform(-1.5, 1.5))
            floor_outside = outside + float(self.np_random.uniform(-2, 2))
            solar = float(self.np_random.uniform(100, 550))

        self.env_values = np.asarray(
            [outside, inside, floor_inside, floor_outside, solar],
            dtype=np.float32
        )

        current = self._current_simulated_row()
        self.history_rows = [current.copy() for _ in range(self.max_lag)]

    def step(self, action):
        action = int(action)
        before_temp = float(self.env_values[1])
        before_error = abs(before_temp - self.target_temp)

        self._advance_state(action)
        self.idx += 1
        self._append_current_simulated_row()

        after_temp = float(self.env_values[1])
        after_error = abs(after_temp - self.target_temp)

        reward, reward_parts = self._reward(action, before_temp, after_temp, before_error, after_error)

        self.previous_action = action
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.episode_len

        return self._normalized_observation(), reward, terminated, truncated, {
            "inside_temp_before": before_temp,
            "inside_temp_after": after_temp,
            "action_name": ACTION_NAMES[action],
            **reward_parts,
        }

    def _advance_state(self, action):
        current_row = self._current_simulated_row()
        lag_row = build_lag_features_from_rows(self.history_rows + [current_row], self.lag_features)
        action_values = action_to_one_hot([action])[0]
        model_input = np.concatenate([lag_row.to_numpy(dtype=np.float32)[0], action_values]).reshape(1, -1)
        model_input_s = self.x_scaler.transform(model_input)

        with torch.no_grad():
            delta_s = self.dynamics(torch.tensor(model_input_s, dtype=torch.float32)).numpy()

        delta = self.y_scaler.inverse_transform(delta_s)[0]
        self.env_values = self.env_values + delta.astype(np.float32)
        self._apply_physics_action_effect(action)
        self._apply_next_weather()

    def _apply_physics_action_effect(self, action):
        bits = ACTION_BITS[int(action)]
        fan_on = int(bits["fan_on"])
        heater_on = int(bits["heater_on"])

        if heater_on:
            self.env_values[1] += HEATER_TEMP_GAIN
            self.env_values[2] += HEATER_FLOOR_GAIN

        if fan_on:
            self.env_values[1] -= FAN_TEMP_DROP
            self.env_values[2] -= FAN_FLOOR_DROP

    def _apply_next_weather(self):
        next_idx = self.idx + 1
        self.env_values[0] = float(self.df.loc[next_idx, "T_outside"])
        self.env_values[3] = float(self.df.loc[next_idx, "T_floor_outside"])
        self.env_values[4] = float(self.df.loc[next_idx, "SR_direct_outside"])

    def _reward(self, action, before_temp, after_temp, before_error, after_error):
        bits = ACTION_BITS[int(action)]
        fan_on = int(bits["fan_on"])
        heater_on = int(bits["heater_on"])

        both_off = fan_on == 0 and heater_on == 0
        both_on = fan_on == 1 and heater_on == 1

        low = self.target_temp - self.comfort_band
        high = self.target_temp + self.comfort_band

        cold_before = before_temp < low
        hot_before = before_temp > high
        comfortable_before = not cold_before and not hot_before

        action_score = 0.0
        severity = 0.0

        if comfortable_before:
            if both_off:
                action_score += COMFORT_BOTH_OFF_BONUS
            elif both_on:
                action_score -= BOTH_ON_PENALTY
            else:
                action_score -= COMFORT_ACTIVE_PENALTY

            after_in_band = low <= after_temp <= high

            comfort_penalty = 0.0 if after_in_band else 20.0 * abs(after_temp - self.target_temp)
            energy_penalty = HEATER_ENERGY_PENALTY * heater_on + FAN_ENERGY_PENALTY * fan_on
            both_on_penalty = BOTH_ON_PENALTY if both_on else 0.0
            switching_penalty = SWITCHING_PENALTY if action != self.previous_action else 0.0

            reward = (
                action_score
                - comfort_penalty
                - energy_penalty
                - both_on_penalty
                - switching_penalty
            )

            return float(reward), {
                "action_score": float(action_score),
                "comfort_penalty": float(comfort_penalty),
                "improvement_reward": 0.0,
                "wrong_direction_penalty": 0.0,
                "energy_penalty": float(energy_penalty),
                "both_on_penalty": float(both_on_penalty),
                "switching_penalty": float(switching_penalty),
            }

        if cold_before:
            severity = low - before_temp

            if action == 1:
                action_score += RIGHT_ACTION_BONUS + 8.0 * severity
            elif action == 0:
                action_score -= WRONG_ACTION_PENALTY + 8.0 * severity
            elif action == 2:
                action_score -= WRONG_ACTION_PENALTY + 12.0 * severity
            elif action == 3:
                action_score -= BOTH_ON_PENALTY + 4.0 * severity

        elif hot_before:
            severity = before_temp - high

            if action == 2:
                action_score += RIGHT_ACTION_BONUS + 8.0 * severity
            elif action == 0:
                action_score -= WRONG_ACTION_PENALTY + 8.0 * severity
            elif action == 1:
                action_score -= WRONG_ACTION_PENALTY + 12.0 * severity
            elif action == 3:
                action_score -= BOTH_ON_PENALTY + 4.0 * severity

        temp_error_after = after_temp - self.target_temp
        outside_band_after = max(abs(temp_error_after) - self.comfort_band, 0.0)

        comfort_penalty = 2.0 * (outside_band_after ** 2)
        improvement_reward = 4.0 * (before_error - after_error)

        wrong_direction_penalty = 0.0

        if cold_before and after_temp < before_temp:
            wrong_direction_penalty += 50.0 + 3.0 * severity

        if hot_before and after_temp > before_temp:
            wrong_direction_penalty += 50.0 + 3.0 * severity

        energy_penalty = HEATER_ENERGY_PENALTY * heater_on + FAN_ENERGY_PENALTY * fan_on
        both_on_penalty = BOTH_ON_PENALTY if both_on else 0.0
        switching_penalty = SWITCHING_PENALTY if action != self.previous_action else 0.0

        reward = (
            action_score
            + improvement_reward
            - comfort_penalty
            - wrong_direction_penalty
            - energy_penalty
            - both_on_penalty
            - switching_penalty
        )

        return float(reward), {
            "action_score": float(action_score),
            "comfort_penalty": float(comfort_penalty),
            "improvement_reward": float(improvement_reward),
            "wrong_direction_penalty": float(wrong_direction_penalty),
            "energy_penalty": float(energy_penalty),
            "both_on_penalty": float(both_on_penalty),
            "switching_penalty": float(switching_penalty),
        }
    def _compose_raw_observation(self, env_values, time_values, previous_action):
        temp_error = float(env_values[1]) - self.target_temp
        below_band = max((self.target_temp - self.comfort_band) - float(env_values[1]), 0.0)
        above_band = max(float(env_values[1]) - (self.target_temp + self.comfort_band), 0.0)
        return np.concatenate([
            np.asarray(env_values, dtype=np.float32),
            np.asarray(time_values, dtype=np.float32),
            np.asarray([float(previous_action), temp_error, below_band, above_band], dtype=np.float32),
        ]).astype(np.float32)

    def _raw_observation(self):
        time_values = self.df.loc[self.idx, ["hour_sin", "hour_cos", "month_sin", "month_cos"]].to_numpy(dtype=np.float32)
        return self._compose_raw_observation(self.env_values, time_values, self.previous_action)

    def _normalized_observation(self):
        raw = self._raw_observation()
        return ((raw - self.obs_mean) / self.obs_std).astype(np.float32)

    def _row_from_df(self, idx):
        row = self.df.loc[idx, SUPERVISED_FEATURES].to_dict()
        return {key: float(value) for key, value in row.items()}

    def _current_simulated_row(self):
        time_values = self.df.loc[self.idx, ["hour_sin", "hour_cos", "month_sin", "month_cos"]].to_dict()
        row = {col: float(value) for col, value in zip(ENV_COLUMNS, self.env_values)}
        row.update({key: float(value) for key, value in time_values.items()})
        return row

    def _append_current_simulated_row(self):
        self.history_rows.append(self._current_simulated_row())
        self.history_rows = self.history_rows[-self.max_lag:]
