# Medical UAV Delivery Environment

A Gymnasium environment in which one drone picks up and delivers medical orders between a medical
center and five hospitals over one simulated day. Orders arrive at random and have deadlines; the
drone must avoid no-fly zones, carry at most 3 boxes and 2 kg, and never run out of battery. Three
baseline policies are included: random, earliest deadline first (EDF) and Slack-aware.

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). Milestone 1: the environment.

![Medical UAV delivery simulation over one day](assets/demo.gif)

## Requirements

- Conda (Miniconda or Anaconda) and an internet connection for the installation.
- Python 3.11.17 with numpy 2.4.6, gymnasium 1.4.0, matplotlib 3.11.2, imageio 2.38.1,
  imageio-ffmpeg 0.6.0 (animation export; it includes FFmpeg), pytest 9.1.1 and
  PyTorch 2.10.0+cpu. `environment.yml` installs all of them.
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

Micromamba can also create the same environment without changing the system Python:

```bash
micromamba create -y -f environment.yml
micromamba run -n medical-uav-rl python -m pip install -e .
micromamba run -n medical-uav-rl python -m pytest -q
```

`requirements-lock.txt` records all resolved Python package versions from the
verified Linux CPU environment. After creating the Python environment, use
`python -m pip install -r requirements-lock.txt` to restore those transitive
dependency versions, then `python -m pip install -e .` to install this project.

## Neural network installation demo

```bash
python scripts/demo_network.py
```

The demo uses the shared `403 -> 256 -> 128` MLP with separate 26-action and
scalar-value heads (139,803 parameters). It collects 256 real environment transitions,
applies each observation's legal-action mask, performs one actor-critic optimizer
step, checks nonzero gradients and changed weights in both heads, and saves/reloads
a checkpoint with exactly matching outputs. It prints input/output shapes and timings;
the JSON record and checkpoint are in `outputs/network_smoke/`.

This is a CPU forward/backward and checkpoint demonstration, **not a PPO trainer**
or evidence of improved scheduling. It uses only Train demand seeds. The default
is one PyTorch CPU thread to avoid excessive thread overhead for this small model;
`--threads`, `--batch-size`, `--seed` and `--output-dir` are configurable.

The pinned PyTorch package is the official CPU build, so this environment does not
require CUDA or a working NVIDIA driver. GPU training can be configured separately
after the driver is working and CPU/GPU throughput has been compared.

## Neural Network C&R submission checklist

- One ZIP/RAR/7z per group containing commented source, a runnable network demo,
  this README and `environment.yml`, plus the group's two-page report.
- The report must explain the main code logic and organization, disclose external
  code/libraries/tools and adaptations, and include reflections/lessons learned
  and any changes to the proposal or remaining work.
- Acknowledge AI use explicitly, describing the tools and their actual uses. The
  group remains responsible for the submitted content and results.
- The supplied assignment screenshot shows Friday 30 October 2026 at 23:59 as the
  deadline. Verify the course portal's displayed timezone before uploading.

The environment and network smoke test cover only part of this deliverable; the
two-page report and full learning pipeline still need to be completed.

## Run the demo

```bash
python scripts/demo.py
```

This simulates one day with the Slack-aware policy (medium load, demand seed 0) and prints the
results, which start with:

```
Completed at 17:05 (t=545.8 min) after 113 decisions
  Orders   70 released: delivered 53/70 (75.7%), expired 17/70 (24.3%)
```

It then saves two files in `outputs/` (about one minute):

- `demo_slack_medium_seed0_t182min.png`, a snapshot at 11:02
- `demo_slack_medium_seed0.gif`, the whole day

| Option | Values |
|---|---|
| `--policy` | `slack` (default), `edf`, `random` |
| `--load` | `low`, `medium` (default), `high` |
| `--seed` | which day of orders (default `0`) |

`scripts/demo.py` is also the usage example: it creates the environment, starts a day, lets a
policy choose actions until the day is over, and then draws the day.

## The environment

| | |
|---|---|
| Observation | `Box(403)`: the drone, up to 12 waiting orders (those that can still be on time first) and the cargo on board |
| Action | `Discrete(26)`: pick up, deliver, charge, return to the center, wait; `info["action_mask"]` marks the allowed ones |
| Reward | +1 per on-time delivery, less when late, −0.5 per expired order, a small energy cost |
| Episode | one day, about 80–120 decisions |
| `reset` options | `demand_seed` (which day of orders), `load` (`"low"`, `"medium"` or `"high"`) |
| Baselines | `random_policy`, `edf_policy`, `slack_aware_policy` in `medical_uav.baselines.heuristics` |

Most settings (map, drone, demand, rewards) are in `medical_uav/env/config.py`; pass
`MedicalUAVEnv(config=EnvConfig(...))` to change them. The order types, the 12 slots and the charge
targets are fixed because they define the observation and action layout. The rules of the
environment are described in the report, `ME5418_Group17_M1_Report.pdf`.

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
  demo.py            run and show one day; also the usage example
assets/
  demo.gif           the default demo day (the animation at the top of this page)
  demo.png           its frame at 11:02 (Figure 2 of the report)
ME5418_Group17_M1_Report.pdf   the two-page report
environment.yml      conda environment
requirements.txt     the same packages for pip
pyproject.toml       package definition for `pip install -e .`
```
