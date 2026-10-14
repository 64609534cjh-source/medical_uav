# Medical UAV Delivery Environment

A Gymnasium environment in which one drone picks up and delivers medical orders between a medical
center and five hospitals over one simulated day. Orders arrive at random and have deadlines; the
drone must avoid no-fly zones, carry at most 3 boxes and 2 kg, and never run out of battery.

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). Milestone 1: the environment and
rule-based baseline policies.

![The environment at 11:42 of a simulated day](assets/demo.png)

## Installation

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
pip install -e .
```

Without conda, use Python 3.11 and run `pip install -r requirements.txt` and `pip install -e .`
(same package versions). Tested on Linux.

## Quick start

```bash
python scripts/example.py        # step-by-step walkthrough of the API
python scripts/demo.py --live    # watch one day (saves a GIF when there is no display)
python -m pytest -q              # run the tests
```

## Usage

```python
from medical_uav.env import MedicalUAVEnv

env = MedicalUAVEnv(load="medium", render_mode="rgb_array")
obs, info = env.reset(options={"demand_seed": 0})   # the same seed gives the same day

done = False
while not done:
    # your policy goes here; this one picks a random allowed action
    action = env.action_space.sample(mask=info["action_mask"].astype("int8"))
    obs, reward, done, truncated, info = env.step(action)

frame = env.render()   # RGB image of the current state
```

| | |
|---|---|
| Observation | `Box(415)`: the drone, the 12 most urgent waiting orders and the cargo on board |
| Action | `Discrete(26)`: pick up, deliver, charge, return to the center, wait; `info["action_mask"]` marks the allowed ones |
| Reward | +1 per on-time delivery, less when late, −0.5 per expired or rejected order, a small energy cost |
| Episode | one day, about 100 decisions |
| `reset` options | `demand_seed` (which day of orders), `load` (`"low"`, `"medium"` or `"high"`) |
| Baselines | `random_policy`, `edf_policy`, `slack_aware_policy` in `medical_uav.baselines.heuristics` |

All settings (map, drone, orders, rewards) are in `medical_uav/env/config.py`; pass
`MedicalUAVEnv(config=EnvConfig(...))` to change them. The rules of the environment are described
in the Milestone 1 report.

## Project structure

```
medical_uav/
  env/
    config.py        settings
    physics.py       flight time, energy, charging time, delivery reward
    astar.py         shortest path around no-fly zones
    map_gen.py       random map generation
    orders.py        orders and the order stream of one day
    simulator.py     simulation of one day
    actions.py       the 26 actions
    masks.py         which actions are allowed
    observation.py   the observation vector
    uav_env.py       MedicalUAVEnv (Gymnasium environment)
    render.py        drawing and animations
  baselines/
    heuristics.py    Random, EDF and Slack-aware policies
  metrics.py         per-day metrics
  splits.py          seeds for training, development and testing
scripts/
  example.py         API walkthrough
  demo.py            watch or record a day
  calibrate.py       load calibration and benchmarks
tests/               pytest suite
```
