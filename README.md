# Medical UAV Delivery: Environment and Policy Networks

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). **Milestone 2: the neural network.**

One drone picks up and delivers medical orders between a medical center and five hospitals over
a simulated day (a Gymnasium environment, built in Milestone 1). Orders arrive at random and have
deadlines; the drone must avoid no-fly zones, carry at most 3 boxes and 2 kg, and never run out of
battery. A policy network chooses the drone's next action at every decision.

## What Milestone 2 contains

1. **Two policy/value networks** (`medical_uav/agents/networks.py`): a set/attention network,
   the design of our proposal, and a plain MLP as a control. Both mask forbidden actions.
2. **The network demo** (`scripts/demo_network.py`): a forward and a backward pass on real
   observations from the environment.
3. **Both networks trained by imitation** of our strongest rule, Slack-aware
   (`medical_uav/agents/bc.py`, `scripts/train_bc.py`); the weights are in `checkpoints/`.
4. **A visualisation of the network's decisions** (`assets/`): a replay web page and two videos,
   paced so each decision can be read (the Milestone 1 animation was too fast to follow).
5. **Environment changes since Milestone 1**: a new reward, travel times in the observation, two
   more rule baselines and an evaluator that compares policies on the same days.

The two-page report `ME5418_Group17_M2_Report.pdf` explains the design and the results.

![A decision of the set network: charge first or pick up medicine, 50.03 % against 49.97 %](assets/demo.png)

## Installation

Needs Conda and an internet connection; no GPU. Tested on Linux. Unzip the submission, open a
terminal (Anaconda Prompt on Windows) in the folder with `environment.yml`, and run:

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
python -m pip install -e .
```

This installs Python 3.11 with numpy 2.4.6, gymnasium 1.4.0, matplotlib 3.11.2, imageio 2.38.1,
imageio-ffmpeg 0.6.0 and torch 2.14.1 (CPU). If `medical-uav-rl` already exists, add
`-n <another name>` to the first command and activate that name. Without conda: Python 3.11,
`python -m pip install -r requirements.txt`, then `python -m pip install -e .`.

## 1. The network demo: forward and backward pass

```bash
python scripts/demo_network.py
```

It prints (no files are saved; about 15 s):

1. both networks and their sizes (MLP 370,459 parameters, set network 62,928);
2. the input: 64 real observations `[64, 452]` with their action masks `[64, 26]`;
3. a **forward pass** of the trained set network: logits `[64, 26]` and values `[64]`, and three
   of its decisions next to Slack-aware's;
4. a **backward pass** of a fresh set network: policy and value losses, gradient norms per
   module, and one Adam step after which the loss falls;
5. a whole day driven by each trained network.

The inputs are Slack-aware's first 64 decisions of a day. Imitation trains only the policy, so the
value head is still untrained.

```
3. forward pass (trained set network): logits (64, 26), value (64,)
   decision 30: PICK sample H3->M 0.83, DELIVER(M) 0.12, DELIVER(H1) 0.05   (Slack-aware chose PICK sample H3->M)
4. backward pass (untrained set network): policy loss 1.849, value loss 13.408
   after one Adam step the loss falls from 8.553 to 8.509
5. A whole day (medium load, demand seed 100000) driven by each trained network
   BC mlp network (validation accuracy 0.935): 95.9% on time
   BC set network (validation accuracy 0.939): 91.8% on time
```

## 2. The visualisation (`assets/`, nothing to install)

All files show one medium-load day never used in training (demand seed 100000), driven by the
trained set network (`replay_slack.html`: by its teacher). Because the Milestone 1 animation was
too fast, the video is slowed down so each decision can be read without touching anything, and
the web page lets you explore the whole day at your own pace.

| File | What it is for |
|---|---|
| `set_network_explained.mp4` (3:23) | **Watch first: the decisions, slowed down.** 29 decisions around the afternoon peak, 13:06–15:30; each takes 7 s (2 s before, 3 s during and 2 s after its action) with the situation and the network's most likely actions on screen. |
| `replay_set_network.html` | **Explore the whole day yourself.** Download it and double-click: it opens offline in any browser. You choose the speed, stop at any decision and jump to any time; every decision shows the situation and up to three of the network's most likely actions (the drone takes the most likely one). It opens at the network's closest call (14:30). |
| `replay_slack.html` | The same page for the teacher, Slack-aware, on the same day (47 of 49 orders on time; the network 45), to compare the two. |
| `set_network_day.mp4` (6.5 min) | An overview of the whole day without clicking: flights at 1 simulated minute per second, the rest at 8, a 1.5 s pause at each decision. |
| `demo.png` | The picture above: the closest call in the replay. |

Using the replay:

- **Speed.** **Play** (or the space bar) uses the selected speed, initially **Smart**: flights at
  1 simulated minute per second, loading, charging and waiting at 8, and a 3 s pause at each
  decision. **1**, **5** and **15** play everything at that many simulated minutes per second,
  with a 1 s pause at each decision.
- **Decision by decision.** Tick **Stop at each decision**, or use **Previous / Next decision**
  (← →). Drag the timeline or use **Jump to** (start of the day, closest call, ...) to move.
- **Reading it.** **How to read this screen**, at the top, explains the map, panels and timeline.
  The percentages are the network's probabilities of choosing each action, not chances of success.

To make these files yourself (they go to `outputs/`; `assets/` is not touched):

```bash
python scripts/demo.py --policy set --seed 100000           # replay and picture, a few seconds
python scripts/demo.py --policy set --seed 100000 --video   # also the two videos, 15-20 minutes
python scripts/demo.py --policy slack --seed 100000         # the teacher's replay
```

`scripts/demo.py` is also the usage example of the environment: it creates it, runs one day with a
policy (`--policy slack|edf|random|mlp|set`, `--load low|medium|high`, `--seed <day>`) and shows
the day.

## Other commands

```bash
python scripts/train_bc.py      # retrain both networks by imitation (about a minute)
python scripts/evaluate.py --methods random,edf,fcfs,nearest,slack --checkpoints checkpoints/bc_mlp.pt checkpoints/bc_set.pt
```

`evaluate.py` runs policies and trained networks on the same days and prints on-time rates with
paired 95 % intervals, as in the report's Table 1.

## Environment and networks

| Environment (`MedicalUAVEnv`) | |
|---|---|
| Observation | `Box(452)`: the drone, travel times between all nodes, up to 12 waiting orders (those that can still be on time first) and the cargo on board |
| Action | `Discrete(26)`: pick up (12 slots), deliver (6), charge (6), return to the center, wait; `info["action_mask"]` marks the allowed ones |
| Reward | +1 per on-time delivery, +0.1 per late one, −0.1 per expired order, −0.001 per battery % used |
| Episode | one day, about 80–120 decisions; `reset(options={"demand_seed": ..., "load": ...})` |

| Networks | MLP | Set network |
|---|---|---|
| Structure | separate actor and critic, 452 → 256 → 256 (tanh) | shared encoders for order and cargo slots, attention from the drone's state, one shared scorer for every pick-up slot |
| Output | logits `[B, 26]` (forbidden actions masked) and a value `[B]` | the same |
| Parameters | 370,459 | 62,928 |

## Files

```
medical_uav/env/        the environment (Milestone 1, updated): uav_env.py, simulator.py, masks.py,
                        observation.py, ...; replay.py, replay_player.html and video.py draw a day
medical_uav/agents/     networks.py (both networks), policy.py (a network as a policy, checkpoints),
                        bc.py (imitation learning)
medical_uav/baselines/  heuristics.py: Random, EDF, FCFS, nearest-first, Slack-aware
medical_uav/            evaluation.py, metrics.py, splits.py (which days are for what)
scripts/                demo_network.py, demo.py, train_bc.py, evaluate.py
checkpoints/            bc_mlp.pt, bc_set.pt (trained networks), bc_history.json (training curves)
assets/                 the replays, videos and picture above
environment.yml, requirements.txt, pyproject.toml
ME5418_Group17_M2_Report.pdf
```
