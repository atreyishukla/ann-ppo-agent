DATA_PATH = "data/Concrete floor results_revised.xlsx"
SHEET_NAME = "Results_1hr"
MODEL_DIR = "models"

ENV_COLUMNS = [
    "T_outside",
    "T_inside",
    "T_floor_inside",
    "T_floor_outside",
    "SR_direct_outside",
]

TIME_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

SUPERVISED_FEATURES = ENV_COLUMNS + TIME_COLUMNS

ACTION_BITS = {
    0: {"fan_on": 0, "heater_on": 0},
    1: {"fan_on": 0, "heater_on": 1},
    2: {"fan_on": 1, "heater_on": 0},
    3: {"fan_on": 1, "heater_on": 1},
}

ACTION_NAMES = {
    0: "fan_off_heater_off",
    1: "fan_off_heater_on",
    2: "fan_on_heater_off",
    3: "fan_on_heater_on",
}

NUM_ACTIONS = 4

DEFAULT_TARGET_TEMP = 22.0
DEFAULT_COMFORT_BAND = 1.0

# Physics-informed action effects inside the training simulator.
# These are not final-output rules. They make the RL environment causal enough
# for the agent to learn that heating raises temperature and fan/cooling lowers it.
HEATER_TEMP_GAIN = 2.20
HEATER_FLOOR_GAIN = 0.60
FAN_TEMP_DROP = 2.20
FAN_FLOOR_DROP = 0.45

BOTH_ON_PENALTY = 120.0
SWITCHING_PENALTY = 0.02

HEATER_ENERGY_PENALTY = 0.20
FAN_ENERGY_PENALTY = 0.18

RIGHT_ACTION_BONUS = 60.0
WRONG_ACTION_PENALTY = 120.0

COMFORT_BOTH_OFF_BONUS = 150.0
COMFORT_ACTIVE_PENALTY = 140.0
