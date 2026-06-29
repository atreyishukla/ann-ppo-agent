import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ENV_COLUMNS, NUM_ACTIONS, SUPERVISED_FEATURES, TIME_COLUMNS


def load_sheet(path, sheet_name="Results_1hr"):
    df = pd.read_excel(path, sheet_name=sheet_name, header=1, usecols="A:H")
    df = df.dropna(subset=["Date_time"]).copy()
    df["Date_time"] = pd.to_datetime(df["Date_time"])
    return df


def add_action_labels(df):
    df = df.copy()
    df["heater_on"] = (df["Heating_power"] > 0).astype(int)
    df["fan_on"] = (df["Cooling_power"] > 0).astype(int)

    # Four-state action encoding:
    # 0 = fan off, heater off
    # 1 = fan off, heater on
    # 2 = fan on, heater off
    # 3 = fan on, heater on
    df["action"] = df["heater_on"] + 2 * df["fan_on"]
    return df


def add_time_features(df):
    df = df.copy()
    hour = df["Date_time"].dt.hour
    month = df["Date_time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


def load_prepared_frame(path, sheet_name="Results_1hr"):
    df = load_sheet(path, sheet_name)
    df = add_action_labels(df)
    df = add_time_features(df)
    df = df.dropna(subset=SUPERVISED_FEATURES + ["action"]).reset_index(drop=True)
    return df


def make_supervised_dataset(path, sheet_name="Results_1hr"):
    df = load_prepared_frame(path, sheet_name)
    X = df[SUPERVISED_FEATURES].astype("float32")
    y = df["action"].astype("int64")
    return df, X, y


def add_lag_features(df, lags=(1, 3, 6, 12, 24)):
    df = df.copy()
    for lag in lags:
        for col in ENV_COLUMNS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"T_inside_delta_{lag}"] = df["T_inside"] - df["T_inside"].shift(lag)
    df["SR_direct_outside_roll_6"] = df["SR_direct_outside"].rolling(6).mean()
    df["T_outside_roll_6"] = df["T_outside"].rolling(6).mean()
    return df


def make_lag_dataset(path, sheet_name="Results_1hr", lags=(1, 3, 6, 12, 24)):
    df = load_prepared_frame(path, sheet_name)
    df = add_lag_features(df, lags)
    lag_features = list(SUPERVISED_FEATURES)
    for lag in lags:
        lag_features.extend([f"{col}_lag_{lag}" for col in ENV_COLUMNS])
        lag_features.append(f"T_inside_delta_{lag}")
    lag_features.extend(["SR_direct_outside_roll_6", "T_outside_roll_6"])
    df = df.dropna(subset=lag_features + ["action"]).reset_index(drop=True)
    X = df[lag_features].astype("float32")
    y = df["action"].astype("int64")
    return df, X, y, lag_features


def make_sequence_dataset(path, sheet_name="Results_1hr", window=24):
    df = load_prepared_frame(path, sheet_name)
    feature_values = df[SUPERVISED_FEATURES].to_numpy(dtype=np.float32)
    labels = df["action"].to_numpy(dtype=np.int64)
    sequences = []
    targets = []
    for i in range(window - 1, len(df)):
        sequences.append(feature_values[i - window + 1:i + 1])
        targets.append(labels[i])
    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    return df.iloc[window - 1:].reset_index(drop=True), X, y, list(SUPERVISED_FEATURES)


def stratified_three_way_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


def action_to_one_hot(actions, num_actions=NUM_ACTIONS):
    actions = np.asarray(actions, dtype=np.int64)
    out = np.zeros((len(actions), num_actions), dtype=np.float32)
    out[np.arange(len(actions)), actions] = 1.0
    return out


def build_lag_features_from_rows(rows, features):
    hist_df = pd.DataFrame(rows)
    base = hist_df.iloc[-1].to_dict()
    output = {feature: base.get(feature, np.nan) for feature in features}

    for feature in features:
        if "_lag_" in feature:
            col, lag = feature.rsplit("_lag_", 1)
            lag = int(lag)
            output[feature] = hist_df[col].iloc[-lag - 1] if len(hist_df) > lag else hist_df[col].iloc[0]
        elif feature.startswith("T_inside_delta_"):
            lag = int(feature.rsplit("_", 1)[-1])
            old_value = hist_df["T_inside"].iloc[-lag - 1] if len(hist_df) > lag else hist_df["T_inside"].iloc[0]
            output[feature] = hist_df["T_inside"].iloc[-1] - old_value
        elif feature == "SR_direct_outside_roll_6":
            output[feature] = hist_df["SR_direct_outside"].tail(6).mean()
        elif feature == "T_outside_roll_6":
            output[feature] = hist_df["T_outside"].tail(6).mean()

    return pd.DataFrame([output], columns=features).astype("float32")


def make_dynamics_dataset(path, sheet_name="Results_1hr"):
    df, X_lag, _, lag_features = make_lag_dataset(path, sheet_name)
    current = df.iloc[:-1].copy().reset_index(drop=True)
    nxt = df.iloc[1:].copy().reset_index(drop=True)
    base_X = X_lag.iloc[:-1].to_numpy(dtype=np.float32)
    action_one_hot = action_to_one_hot(current["action"].to_numpy())
    X = np.concatenate([base_X, action_one_hot], axis=1).astype(np.float32)
    y = (nxt[ENV_COLUMNS].to_numpy() - current[ENV_COLUMNS].to_numpy()).astype("float32")
    return current, X, y, lag_features


def build_time_features(timestamp):
    timestamp = pd.to_datetime(timestamp)
    hour = timestamp.hour
    month = timestamp.month
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }


def payload_to_feature_dict(payload):
    row = {col: float(payload[col]) for col in ENV_COLUMNS}
    row.update(build_time_features(payload["Date_time"]))
    return row


def build_current_feature_row(payload):
    row = payload_to_feature_dict(payload)
    return pd.DataFrame([row], columns=SUPERVISED_FEATURES).astype("float32")


def build_lag_feature_row(payload, features):
    history = payload.get("history") or []
    rows = [payload_to_feature_dict(item) for item in history]
    rows.append(payload_to_feature_dict(payload))
    return build_lag_features_from_rows(rows, features)


def build_sequence_array(payload, features, window):
    history = payload.get("history") or []
    rows = [payload_to_feature_dict(item) for item in history]
    rows.append(payload_to_feature_dict(payload))
    if len(rows) < window:
        rows = [rows[0]] * (window - len(rows)) + rows
    rows = rows[-window:]
    X = pd.DataFrame(rows, columns=features).astype("float32").to_numpy()
    return X.reshape(1, window, len(features))
