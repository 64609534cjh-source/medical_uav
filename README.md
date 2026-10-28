# Medical UAV Delivery: Environment and Policy Networks

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). **Milestone 2: the neural network.**

A drone picks up and delivers medical orders between a medical center and five hospitals over a
simulated day (our Milestone 1 Gymnasium environment). A policy network chooses its next action
at every decision.

## What Milestone 2 contains

1. **Two policy/value networks** (`medical_uav/agents/networks.py`): a set/attention network
   (our proposal's design) and a plain MLP as a control, both with action masking.
2. **The network demo** (`scripts/demo_network.py`): a forward and a backward pass.
3. **Both networks trained by imitation** of our best rule, Slack-aware (`bc.py`,
   `scripts/train_bc.py`); weights in `checkpoints/`.
4. **A visualisation of the network's decisions** (`assets/`), paced so each decision can be read.
5. **Environment updates**: new reward, travel times in the observation, two more rule baselines,
   an evaluator.

Design and results are in the two-page report `ME5418_Group17_M2_Report.pdf`.

![A decision of the set network: charge first or pick up medicine, 50.03 % against 49.97 %](assets/demo.png)

## Installation

Needs Conda; no GPU; tested on Linux. `environment.yml` pins Python 3.11, PyTorch 2.14.1 (CPU) and
the other libraries. In a terminal in the unzipped folder:

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
python -m pip install -e .
```

If `medical-uav-rl` already exists, add `-n <another name>` to the first command. Without conda:
Python 3.11, `pip install -r requirements.txt`, then `pip install -e .`.

## 1. The network demo: forward and backward pass

```bash
python scripts/demo_network.py
```

Prints in about 15 s (no files saved):

1. both networks and their sizes;
2. the input: 64 real observations `[64, 452]` and action masks `[64, 26]`;
3. **forward pass** of the trained set network: logits `[64, 26]`, values `[64]`, three decisions
   next to Slack-aware's;
4. **backward pass** of a fresh set network: losses, gradient norms, one Adam step (loss falls);
5. a whole day driven by each trained network.

```
3. forward pass (trained set network): logits (64, 26), value (64,)
   decision 30: PICK sample H3->M 0.83, DELIVER(M) 0.12, DELIVER(H1) 0.05   (Slack-aware chose PICK sample H3->M)
4. backward pass (untrained set network): policy loss 1.849, value loss 13.408
   after one Adam step the loss falls from 8.553 to 8.509
```

The value head is untrained: imitation trains only the policy.

## 2. The visualisation (`assets/`, nothing to install)

One unseen medium-load day (demand seed 100000), driven by the trained set network. Pauses at
decisions are only for reading; the network decides in about 1 ms.

| File | What it is for |
|---|---|
| `set_network_explained.mp4` (3:23) | **Watch first.** 29 decisions of the afternoon peak, slowed down: 7 s each, with the situation and the network's top choices. |
| `replay_set_network.html` | **Explore the whole day.** Download and open in a browser (offline). Pause, step through decisions, change speed. |
| `replay_slack.html` | The same day driven by the teacher, Slack-aware, for comparison. |
| `set_network_day.mp4` (6.5 min) | The whole day as an overview video. |

Replay controls: **Smart** speed (default) plays flights and box handling at 1 simulated minute
per second, charging and waiting at 3, and pauses 3 s at each decision; **1 / 5 / 15** are fixed
speeds. **Stop at each decision** or **Previous / Next decision** (← →) step one decision at a
time. Percentages are the network's action probabilities; the drone takes the highest.

To regenerate them in `outputs/`: `python scripts/demo.py --policy set --seed 100000 [--video]`
(`--policy slack` for the teacher). `scripts/demo.py` is also the environment's usage example.

## Other commands

```bash
python scripts/train_bc.py      # retrain both networks by imitation (about a minute)
python scripts/evaluate.py --methods random,edf,fcfs,nearest,slack --checkpoints checkpoints/bc_mlp.pt checkpoints/bc_set.pt
```

`evaluate.py` compares policies on the same days (the report's Table 1).

## Environment and networks

| Environment | |
|---|---|
| Observation | `Box(452)`: drone, travel times, up to 12 waiting orders, cargo |
| Action | `Discrete(26)`: pick up (12), deliver (6), charge (6), return, wait; `info["action_mask"]` |
| Reward | +1 on time, +0.1 late, −0.1 expired, −0.001 per battery % |
| Episode | one day, 80–120 decisions; `reset(options={"demand_seed": ..., "load": ...})` |

| Networks | MLP | Set network |
|---|---|---|
| Structure | actor and critic, 452 → 256 → 256 | shared order and cargo encoders, attention, shared pick-up scorer |
| Output | logits `[B, 26]` (masked), value `[B]` | the same |
| Parameters | 370,459 | 62,928 |

## Files

```
medical_uav/env/        environment (Milestone 1, updated); replay.py and video.py draw a day
medical_uav/agents/     networks.py, policy.py (checkpoints), bc.py (imitation)
medical_uav/baselines/  rule policies: Random, EDF, FCFS, nearest-first, Slack-aware
medical_uav/            evaluation.py, metrics.py, splits.py
scripts/                demo_network.py, demo.py, train_bc.py, evaluate.py
checkpoints/            trained networks and their training curves
assets/                 replays, videos, picture
```
