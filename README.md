# Medical UAV Delivery: Policy and Value Networks

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). **Milestone 2: the neural network.**

A drone picks up and delivers medical orders between a medical center and five hospitals over a
simulated day (our Milestone 1 Gymnasium environment). A policy network chooses its next action
at every decision. Design and results: the two-page report `ME5418_Group17_M2_Report.pdf`.

## What Milestone 2 contains

1. **Two policy/value networks** (`medical_uav/agents/networks.py`): a set/attention network
   (our proposal's design) and a plain MLP as a control, both with action masking.
2. **The network demo** (`scripts/demo_network.py`): a forward and a backward pass.
3. **Both networks trained by imitation** of our best rule, Slack-aware; weights in `checkpoints/`.
4. **Replays of the network's decisions** (`assets/`), paced so each decision can be read.
5. **Environment updates**: revised reward, travel times in the observation, FCFS and
   nearest-first rules, an evaluator.

![A decision of the set network: charge first or pick up medicine, 50.03 % against 49.97 %](assets/demo.png)

## Install and run the network demo

Needs Conda; no GPU; tested on Linux. `environment.yml` pins the exact environment (Python
3.11.17, PyTorch 2.14.1 CPU and the other libraries). In a terminal in the unzipped folder:

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
python -m pip install -e .
python scripts/demo_network.py
```

(If `medical-uav-rl` already exists, use `-n <name>` in the first command and activate `<name>`.)

In about 15 s the demo prints (no files saved):

1. both networks' sizes, and the input: 64 real observations `[64, 452]` with masks `[64, 26]`;
2. **forward pass** of the trained set network: logits `[64, 26]`, values `[64]`, three
   decisions next to Slack-aware's;
3. **backward pass** of a fresh set network: policy and value losses, gradient norms, one Adam
   step (the loss falls);
4. a whole day driven by each trained network and by the teacher.

Imitation trains only the policy, so the trained networks' value heads are still untrained.

## Replays (`assets/`, nothing to install)

Download a page and open it in a browser (it works offline). Both show the same unseen
medium-load day (demand seed 100000) and open at 09:58, as orders start to pile up.

| File | What it shows |
|---|---|
| [replay_set_network.html](assets/replay_set_network.html) | The trained set network: 45 of 49 orders on time. Each decision shows the situation and the network's top choices with their probabilities; the drone takes the most likely one. |
| [replay_slack.html](assets/replay_slack.html) | Its teacher, Slack-aware, on the same day (47 of 49), for comparison. |
| [demo.png](assets/demo.png) | The picture above: the network's closest call of the day (14:30). |

Press **Play** (or Space). At each decision the clock stops for about 2 s: the choices appear, the
chosen one is highlighted and moves up into the action line, which then shows its progress. These
pauses are only for reading; the network decides in about 1 ms. **Smart** speed (the default)
plays flights at 1 simulated min/s, box handling at 1.5, charging and waiting at 3; **1 / 5 / 15**
are fixed speeds. **Previous / Next decision** (← →) and **Stop at each decision** step through
the day; **How to read this screen** explains the display.

## Other commands

```bash
python scripts/demo.py --policy set --seed 100000           # the replay and picture in outputs/
python scripts/train_bc.py                                  # retrain both networks (~1 min)
python scripts/evaluate.py --checkpoints checkpoints/bc_mlp.pt checkpoints/bc_set.pt
```

`demo.py` is also the environment's usage example (`--policy slack` for the teacher).
`train_bc.py` overwrites `checkpoints/`. `evaluate.py` runs the five rules and both networks on
the same 40 days per load (the report's Table 1) and saves `outputs/evaluation.csv`.

## Environment and networks

| Environment | |
|---|---|
| Observation | `Box(452)`: 80 global features (drone, travel times), 12 waiting orders × 27, 3 cargo slots × 16 |
| Action | `Discrete(26)`: 12 pick up, 6 deliver, 6 charge, return, wait; allowed ones in `info["action_mask"]` |
| Reward | +1 on time, +0.1 late, −0.1 expired, −0.001 per battery % used |
| Episode | one day, low, medium or high demand; `reset(options={"demand_seed": ..., "load": ...})` |

Both networks take observations `[B, 452]` and masks `[B, 26]` and return logits `[B, 26]`
(forbidden actions set to −10⁹) and values `[B]`.

| Network | Structure | Parameters |
|---|---|---:|
| MLP (control) | separate actor and critic, each 452 → 256 → 256, tanh | 370,459 |
| Set/attention | shared order and cargo encoders (width 64), four-head attention, shared pick-up scorer | 62,928 |

## Main files

```
medical_uav/agents/     networks.py (both networks), policy.py (inference, checkpoints), bc.py
medical_uav/env/        the environment; replay.py, replay_player.html, video.py draw a day
medical_uav/baselines/  rule policies: Random, EDF, FCFS, nearest-first, Slack-aware
medical_uav/            evaluation.py, metrics.py, splits.py (which days are for what)
scripts/                demo_network.py, demo.py, train_bc.py, evaluate.py
checkpoints/            bc_set.pt, bc_mlp.pt (weights), bc_history.json (training curves)
assets/                 replays, videos, picture
```
