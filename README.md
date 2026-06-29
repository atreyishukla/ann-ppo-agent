# ANN-PPO Temperature Controller

A local machine learning and reinforcement learning temperature controller for greenhouse-style heating and cooling systems.

This project uses an ANN-lag model to learn temperature-control behavior from historical data and a PPO reinforcement learning agent to make fan/heater control decisions. The controller is designed to run locally and integrate with Node-RED through a custom node.

---

## Project Overview

The goal of this project is to predict and optimize greenhouse heating and cooling decisions using environmental sensor inputs.

Current input variables include:

* `Date_time`
* `T_outside`
* `T_inside`
* `T_floor_inside`
* `T_floor_outside`
* `SR_direct_outside`
* `previous_action`

Current outputs include:

* `fan_on`
* `fan_off`
* `heater_on`
* `heater_off`

The controller uses a four-state action space:

| Action ID | Action              |
| --------: | ------------------- |
|         0 | fan off, heater off |
|         1 | fan off, heater on  |
|         2 | fan on, heater off  |
|         3 | fan on, heater on   |

---

## Architecture

Historical greenhouse data is used to train an ANN-lag model. The ANN-lag approach uses current and previous environmental readings so the model can capture short-term temperature memory.

The reinforcement learning agent is trained using PPO. The PPO agent learns a control policy based on a reward function that considers temperature comfort, energy use, switching behavior, and incorrect actions.

```text
historical data
        ↓
ANN-lag model
        ↓
custom reinforcement learning environment
        ↓
PPO reinforcement learning agent
        ↓
Node-RED custom controller node
        ↓
fan/heater control decision
```

---

## Folder Structure

```text
ann-ppo-agent/
├── data/
│   └── data file goes here
├── models/
│   └── trained model artifacts are saved here
├── node-red/
│   ├── package.json
│   ├── greenhouse-controller.js
│   ├── greenhouse-controller.html
│   └── rl-4-state-flow.json
├── reports/
│   ├── ann_lag_graphs/
│   └── ann_lag_data/
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
│   ├── test_controller_cases.py
│   └── generate_ann_lag_graphs.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/atreyishukla/ann-ppo-agent.git
cd ann-ppo-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Setup

Place the greenhouse data file inside the `data/` folder.

Example:

```text
data/Concrete floor results_revised.xlsx
```

Raw data files are ignored by Git by default, so the dataset must be added locally before training.

---

## Training

Train the ANN-lag supervised model:

```bash
python src/train_ann_lag_only.py
```

Train the ANN-lag dynamics model:

```bash
python src/train_dynamics_model.py
```

Train the PPO reinforcement learning agent:

```bash
python src/train_rl_agent.py --timesteps 500000 --target-temp 22 --comfort-band 1 --episode-len 24 --curriculum-prob 1.0
```

The trained model artifacts are saved in the `models/` folder.

---

## Testing

Run the built-in controller test cases:

```bash
python src/test_controller_cases.py
```

Expected behavior:

* Cold conditions should prefer `fan_off_heater_on`.
* Hot conditions should prefer `fan_on_heater_off`.
* Comfortable conditions should prefer `fan_off_heater_off`.
* `fan_on_heater_on` should generally be discouraged unless specifically needed.

---

## Local Prediction

Run a sample prediction:

```bash
python src/predict_controller.py
```

Example input:

```json
{
  "Date_time": "2026-01-01 08:00:00",
  "T_outside": -8,
  "T_inside": 16,
  "T_floor_inside": 20,
  "T_floor_outside": -6,
  "SR_direct_outside": 100,
  "previous_action": 0
}
```

Example output:

```json
{
  "action_id": 1,
  "action": "fan_off_heater_on",
  "fan_on": 0,
  "fan_off": 1,
  "heater_on": 1,
  "heater_off": 0
}
```

---

## Generating ANN-lag Graphs

Generate ANN-lag performance graphs:

```bash
python src/generate_ann_lag_graphs.py
```

Graphs are saved to:

```text
reports/ann_lag_graphs/
```

Generated CSV files are saved to:

```text
reports/ann_lag_data/
```

The generated graphs include:

* Historical action distribution
* ANN-lag confusion matrix
* ANN-lag F1 score by action
* Actual vs predicted action counts

---

## Node-RED Integration

This project includes a custom Node-RED node for local deployment.

Install the custom node from the project folder:

```bash
cd ~/.node-red
npm install /path/to/ann-ppo-agent/node-red
node-red
```

Then import the Node-RED flow:

```text
node-red/rl-4-state-flow.json
```

In the custom node settings, set:

```text
Project path: /path/to/ann-ppo-agent
Python path: /path/to/ann-ppo-agent/.venv/bin/python
```

Example macOS paths:

```text
Project path: /Users/yourname/Desktop/ann-ppo-agent
Python path: /Users/yourname/Desktop/ann-ppo-agent/.venv/bin/python
```

---

## Model Notes

The ANN-lag model learns from historical control behavior. It is useful for predicting past fan/heater decisions and capturing recent temperature trends.

The PPO reinforcement learning agent is used as the optimization layer. It attempts to learn better control decisions based on reward signals such as comfort, energy use, and action penalties.

The RL agent is still experimental and depends heavily on reward design and the quality of the learned environment model.

---

## Limitations

The dataset is imbalanced. Most historical records are fan off and heater off, while active heating and cooling actions occur less often.

The both-on state is not represented in the available historical data, so it cannot be fully learned from data alone.

The current version does not include humidity, electricity price, crop type, or weather forecasts. These variables could improve future versions of the controller.

---

## Future Improvements

Potential future improvements include:

* adding humidity and weather forecast inputs
* adding electricity cost or energy pricing
* collecting more examples of rare control states
* improving the PPO reward function
* comparing ANN-lag against LSTM for longer sequence memory
* evaluating RL performance over full 24-hour simulations
* deploying the final controller into more Node-RED flows

---

## Git Notes

Raw data files, trained model artifacts, virtual environments, and local dependencies are ignored by Git to keep the repository lightweight.

To reproduce the project, add the data file locally and run the training scripts in order.
