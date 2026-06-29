import json

from predict_rl import predict

TEST_CASES = [
    {
        "name": "very cold morning - should learn heater only",
        "payload": {
            "Date_time": "2026-01-15 07:00:00",
            "T_outside": -18,
            "T_inside": 12,
            "T_floor_inside": 14,
            "T_floor_outside": -10,
            "SR_direct_outside": 20,
            "previous_action": 0,
        },
    },
    {
        "name": "cold night - should learn heater only",
        "payload": {
            "Date_time": "2026-02-02 02:00:00",
            "T_outside": -22,
            "T_inside": 10,
            "T_floor_inside": 12,
            "T_floor_outside": -15,
            "SR_direct_outside": 0,
            "previous_action": 0,
        },
    },
    {
        "name": "hot summer afternoon - should learn fan only",
        "payload": {
            "Date_time": "2026-07-20 15:00:00",
            "T_outside": 31,
            "T_inside": 30,
            "T_floor_inside": 28,
            "T_floor_outside": 29,
            "SR_direct_outside": 750,
            "previous_action": 0,
        },
    },
    {
        "name": "very hot greenhouse - should learn fan only",
        "payload": {
            "Date_time": "2026-08-05 14:00:00",
            "T_outside": 35,
            "T_inside": 36,
            "T_floor_inside": 33,
            "T_floor_outside": 32,
            "SR_direct_outside": 850,
            "previous_action": 0,
        },
    },
    {
        "name": "comfortable - should learn both off",
        "payload": {
            "Date_time": "2026-05-20 10:00:00",
            "T_outside": 21,
            "T_inside": 22,
            "T_floor_inside": 22,
            "T_floor_outside": 20,
            "SR_direct_outside": 250,
            "previous_action": 0,
        },
    },
]


for case in TEST_CASES:
    print("\n" + case["name"])
    print(json.dumps(predict(case["payload"]), indent=2))
