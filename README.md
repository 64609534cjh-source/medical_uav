# Medical UAV Delivery: Environment and Policy Networks

A Gymnasium environment in which one drone picks up and delivers medical orders between a medical
center and five hospitals over one simulated day. Orders arrive at random and have deadlines; the
drone must avoid no-fly zones, carry at most 3 boxes and 2 kg, and never run out of battery.
The package contains rule-based baselines and two policy/value networks (an MLP and a
set/attention network) trained by imitating the strongest rule.

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). Milestone 2: the neural network.

![A decision of the trained set network, explained](assets/demo.png)

Ready-made videos are in `assets/` (see [Videos](#videos)).

## Requirements

- Conda (Miniconda or Anaconda) and an internet connection for the installation.
- Python 3.11 with numpy 2.4.6, gymnasium 1.4.0, matplotlib 3.11.2, imageio 2.38.1,
  imageio-ffmpeg 0.6.0 (video export; it includes FFmpeg) and torch 2.14.1 (CPU build).
  `environment.yml` installs all of them; no GPU is needed.
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

## Run the network demo

```bash
python scripts/demo_network.py
```

It creates both networks, runs a forward pass and a backward pass on real observations from the
environment, and lets the trained networks (in `checkpoints/`) run a whole day. It takes a few
seconds and prints, among other things:

```
3. forward pass (trained set network): logits (64, 26), value (64,)
   decision 30: PICK sample H3->M 0.83, DELIVER(M) 0.12, DELIVER(H1) 0.05   (Slack-aware chose PICK sample H3->M)
      their logits: PICK sample H3->M 15.71, DELIVER(M) 13.74, DELIVER(H1) 12.95;  value estimate -0.237 (untrained: BC does not train the value head)
4. backward pass (untrained set network): policy loss 1.849, value loss 13.408
   after one Adam step the loss falls from 8.553 to 8.509
5. A whole day (medium load, demand seed 100000) driven by each trained network
   BC mlp network (validation accuracy 0.935): 95.9% on time
   BC set network (validation accuracy 0.939): 91.8% on time
   Slack-aware (the teacher) on the same day: 95.9% on time
```

## Run the day demo

```bash
python scripts/demo.py
```

This simulates one day with the Slack-aware policy (medium load, demand seed 0) and prints the
results, which start with:

```
Completed at 17:05 (t=545.8 min) after 113 decisions
  Orders   70 released: delivered 53/70 (75.7%), expired 17/70 (24.3%)
```

It then saves three files in `outputs/` (about two minutes):

- `demo_slack_medium_seed0_t182min.png`, a snapshot at 11:02
- `demo_slack_medium_seed0_explained.mp4`, eight decisions from 11:02, each shown before, during
  and after its action, with the orders it concerns highlighted
- `demo_slack_medium_seed0_day.mp4`, the whole day at 5 simulated minutes per second

| Option | Values |
|---|---|
| `--policy` | `slack` (default), `edf`, `random`, `mlp`, `set` (the trained networks; their top choices are shown in the panel) |
| `--load` | `low`, `medium` (default), `high` |
| `--seed` | which day of orders (default `0`) |
| `--explain-count`, `--day-speed` | decisions in the explanation video (8), simulated minutes per second in the day video (5) |

`scripts/demo.py` is also the usage example: it creates the environment, starts a day, lets a
policy choose actions until the day is over, and then draws the day.

## Videos

| File in `assets/` | What it shows |
|---|---|
| `demo_network_explained.mp4` | the trained set network deciding, eight decisions from 11:02, with its top three choices |
| `demo_network_day.mp4` | the same day driven by the set network, 5 simulated minutes per second |
| `demo_slack_explained.mp4` | the same eight-decision explanation for the Slack-aware rule |

## The environment

| | |
|---|---|
| Observation | `Box(452)`: the drone, travel times between all nodes, up to 12 waiting orders (those that can still be on time first) and the cargo on board |
| Action | `Discrete(26)`: pick up, deliver, charge, return to the center, wait; `info["action_mask"]` marks the allowed ones |
| Reward | +1 per on-time delivery, +0.1 per late delivery, −0.1 per expired order, −0.001 per battery % used |
| Episode | one day, about 80–120 decisions |
| `reset` options | `demand_seed` (which day of orders), `load` (`"low"`, `"medium"` or `"high"`) |
| Baselines | `random_policy`, `edf_policy`, `fcfs_policy`, `nearest_policy`, `slack_aware_policy` in `medical_uav.baselines.heuristics` |

Since Milestone 1 the reward and the observation have changed (environment version `v2`): late
deliveries now earn a small fixed reward instead of a value falling to −0.5, and the observation
adds the 7 × 7 travel-time matrix. Most settings are in `medical_uav/env/config.py`; pass
`MedicalUAVEnv(config=EnvConfig(...))` to change them.

## The networks

| | MLP | Set network |
|---|---|---|
| Input | observation `[B, 452]` and action mask `[B, 26]` | the same, split into the global state, 12 order slots and 3 cargo slots |
| Structure | separate actor and critic, 452 → 256 → 256 (tanh) | shared encoders per slot, one attention layer, one shared scorer for every pick-up slot |
| Output | action logits `[B, 26]` (forbidden actions masked) and a value `[B]` | the same |
| Parameters | 370,459 | 62,928 |

Both were trained by behaviour cloning on 900 days of Slack-aware decisions
(`python scripts/train_bc.py`, about a minute on a CPU). Compare any policies on the same days
with the evaluator, for example:

```bash
python scripts/evaluate.py --methods random,edf,fcfs,nearest,slack --checkpoints checkpoints/bc_mlp.pt checkpoints/bc_set.pt
```

The rules of the environment are described in the Milestone 1 report and the networks in the
two-page report `ME5418_Group17_M2_Report.pdf`.

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
    render.py        drawing, explanation and day videos
  agents/
    networks.py      the MLP and the set/attention network
    policy.py        a network as a policy; saving and loading checkpoints
    bc.py            behaviour cloning: recording a teacher, training a network
  baselines/
    heuristics.py    Random, EDF, FCFS, nearest-first and Slack-aware policies
  evaluation.py      running policies on the same days and comparing them
  metrics.py         per-day metrics
  splits.py          seeds for training, screening, model selection and testing
scripts/
  demo_network.py    network demo: forward and backward pass, a day with each network
  demo.py            run and show one day; also the usage example
  train_bc.py        behaviour cloning of Slack-aware for both networks
  evaluate.py        compare policies on the same days
checkpoints/
  bc_mlp.pt, bc_set.pt  the trained networks; bc_history.json, their training curves
assets/
  demo.png           the set network at 11:02 (the picture at the top of this page)
  *.mp4              the videos listed above
ME5418_Group17_M2_Report.pdf   the two-page report
environment.yml      conda environment
requirements.txt     the same packages for pip
pyproject.toml       package definition for `pip install -e .`
```
