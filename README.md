# ANN-PPO Temperature Controller

This project implements a local machine learning and reinforcement learning temperature controller. The system uses an ANN-lag dynamics model and a PPO reinforcement learning agent to decide whether the greenhouse fan and heater should be on or off.

The controller is designed to run locally and integrate with Node-RED through a custom node.

## Project Goal

The goal is to optimize greenhouse heating and cooling decisions using environmental sensor inputs.

Current inputs include:

- Date_time
- T_outside
- T_inside
- T_floor_inside
- T_floor_outside
- SR_direct_outside
- previous_action

Current outputs include:

- fan_on
- fan_off
- heater_on
- heater_off

The reinforcement learning action space is:

- 0 = fan_off_heater_off
- 1 = fan_off_heater_on
- 2 = fan_on_heater_off
- 3 = fan_on_heater_on

## Architecture

Historical greenhouse data is used to train an ANN-lag dynamics model. That learned dynamics model is then used inside a custom reinforcement learning environment. A PPO agent is trained in this environment to learn a greenhouse control policy.

Architecture:

    historical data
            ↓
    ANN-lag dynamics model
            ↓
    custom greenhouse RL environment
            ↓
    PPO reinforcement learning agent
            ↓
    Node-RED custom controller node
            ↓
    fan/heater control decision

## Folder Structure

    temp-ml-controller/
    ├── data/
    │   └── data file goes here
    ├── models/
    │   └── trained model artifacts are saved here
    ├── node-red/
    │   ├── package.json
    │   ├── greenhouse-controller.js
    │   ├── greenhouse-controller.html
    │   └── rl-4-state-flow.json
    ├── src/
    │   ├── config.py
    │   ├── data_utils.py
    │   ├── models.py
    │   ├── train_ann_lag_only.py
    │   ├── train_dynamics_model.py
    │   ├── train_rl_agent.py
    │   ├── greenhouse_env.py
    │   ├── predict_controller.py
    │   ├── predict_rl.py
    │   └── test_controller_cases.py
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Setup

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Data

Place the Excel data file inside the data folder.

Example:

    data/Concrete floor results_revised.xlsx

Raw data files are ignored by Git by default.

## Training

Train the ANN-lag model:

    python src/train_ann_lag_only.py

Train the dynamics model:

    python src/train_dynamics_model.py

Train the PPO reinforcement learning agent:

    python src/train_rl_agent.py --timesteps 500000 --target-temp 22 --comfort-band 1 --episode-len 24 --curriculum-prob 1.0

Test the controller:

    python src/test_controller_cases.py

Expected behavior:

- Cold cases should prefer heater on.
- Hot cases should prefer fan on.
- Comfortable cases should prefer both fan and heater off.
- Both fan and heater on should generally be discouraged.

## Local Prediction

Run a sample prediction:

    python src/predict_controller.py

Example input:

    {
      "Date_time": "2026-01-01 08:00:00",
      "T_outside": -8,
      "T_inside": 16,
      "T_floor_inside": 20,
      "T_floor_outside": -6,
      "SR_direct_outside": 100,
      "previous_action": 0
    }

Example output:

    {
      "action_id": 1,
      "action": "fan_off_heater_on",
      "fan_on": 0,
      "fan_off": 1,
      "heater_on": 1,
      "heater_off": 0
    }

## Node-RED Integration

Install the custom Node-RED node locally:

    cd ~/.node-red
    npm install /Users/atreyishukla/Desktop/temp-ml-controller/node-red
    node-red

Then import the flow:

    node-red/rl-4-state-flow.json

In the custom node settings, use:

Project path:

    /Users/atreyishukla/Desktop/temp-ml-controller

Python path:

    /Users/atreyishukla/Desktop/temp-ml-controller/.venv/bin/python

## Notes

The trained model files and raw data files are not committed to GitHub by default. This keeps the repository lightweight and avoids uploading local/private data.

To reproduce the model, add the data file locally and rerun the training scripts.
