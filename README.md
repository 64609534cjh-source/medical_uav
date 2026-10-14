# Medical UAV Delivery Environment

A Gymnasium environment in which one drone picks up and delivers medical orders between a medical
center and five hospitals over one simulated day. Orders arrive at random and have deadlines; the
drone must avoid no-fly zones, carry at most 3 boxes and 2 kg, and never run out of battery. Three
baseline policies are included: random, earliest deadline first (EDF) and Slack-aware.

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). Milestone 1: the environment.

![The environment at 11:42 of a simulated day](assets/demo.png)

## Requirements

- Conda (Miniconda or Anaconda) and an internet connection for the installation.
- Python 3.11 with numpy 2.4.6, gymnasium 1.4.0, matplotlib 3.11.2, imageio 2.38.1 and
  imageio-ffmpeg 0.6.0 (animation export; it includes FFmpeg). `environment.yml` installs all
  of them.
- Tested on Linux.

## Installation

Unzip the submission, open a terminal (Anaconda Prompt on Windows) in the folder that contains
`environment.yml`, and run:

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
python -m pip install -e .
```

If an environment named `medical-uav-rl` already exists, add `-n <another name>` to the first
command and activate that name instead. Without conda, use Python 3.11 and run
`python -m pip install -r requirements.txt`, then `python -m pip install -e .`.

## Run the demo

Run all commands from the same folder.

```bash
python scripts/demo.py --no-video
```

This runs one simulated day with the Slack-aware policy (medium load, demand seed 0), prints the
results and saves the frame shown above to `outputs/demo_slack_medium_seed0_t222min.png`. It takes
a few seconds. The output starts with:

```
Medical UAV | Slack-aware | medium load | demand seed 0 | map seed 0

Completed at 16:54 (t=534.1 min) after 111 decisions
  Orders   64 released: delivered 52/64 (81.2%), expired 12/64 (18.8%), rejected 0/64 (0.0%)
  On time  45/64 (70.3%); late 7 (mean 29.7 min late)
```

To see the whole day:

```bash
python scripts/demo.py --live    # play the day in a window; saves a GIF if no window can open
python scripts/demo.py           # save the day as outputs/demo_slack_medium_seed0.gif (about 1 min)
```

| Option | Meaning |
|---|---|
| `--policy {slack,edf,random}` | which baseline policy decides (default `slack`) |
| `--load {low,medium,high}` | how many orders arrive (default `medium`) |
| `--seed N` | which day of orders (default 0) |
| `--snapshots 0 100 ...` | save PNG frames at these simulated minutes (default 222) |
| `--out FILE` | animation file, `.gif` or `.mp4` |

`python scripts/demo.py --help` lists all options.

## Using the environment in code

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
in the report.

## Files

```
medical_uav/
  env/
    uav_env.py       MedicalUAVEnv, the Gymnasium environment
    config.py        settings
    simulator.py     simulation of one day
    actions.py       the 26 actions
    masks.py         which actions are allowed
    observation.py   the observation vector
    map_gen.py       random map generation
    astar.py         shortest path around no-fly zones
    orders.py        orders and the order stream of one day
    physics.py       flight time, energy, charging time, delivery reward
    render.py        drawing and animations
  baselines/
    heuristics.py    Random, EDF and Slack-aware policies
  metrics.py         per-day metrics
  splits.py          seeds for training, development and testing
scripts/
  demo.py            run and show one day
environment.yml      conda environment
requirements.txt     the same packages for pip
pyproject.toml       package definition for `pip install -e .`
```
