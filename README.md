# Medical UAV Delivery: Policy and Value Networks

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). **Milestone 2: neural network code.**

One drone collects and delivers medical orders between a medical center and five hospitals in our
Milestone 1 Gymnasium environment; a policy network chooses each action. Design and results are in
the two-page report `ME5418_Group17_M2_Report.pdf`.

## What Milestone 2 contains

1. **Two policy/value networks** (`medical_uav/agents/networks.py`): the proposed set/attention
   network and an MLP control, both with action masking.
2. **A network demo** (`scripts/demo_network.py`): forward and backward passes on real observations.
3. **Trained checkpoints** (`checkpoints/`): both networks trained by imitation (behaviour cloning)
   of Slack-aware, and a second pair imitating the rollout planner.
4. **A new replay** (`assets/`) that replaces the M1 animation, which was too fast to follow: it
   pauses at each decision to show the network's choices.
5. **Environment updates:** revised reward, travel times in the observation, more baselines and an
   evaluator.

![A decision in the set network's replay](assets/demo.png)

## Install and run the network demo

Requires Conda; no GPU. Tested on Linux. `environment.yml` pins the exact environment (Python
3.11.17, PyTorch 2.14.1 CPU and the other libraries). Open a terminal in the unzipped folder:

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
python -m pip install -e .
python scripts/demo_network.py
```

If `medical-uav-rl` already exists, add `-n medical-uav-m2` to the first command and activate
`medical-uav-m2` instead.

The demo takes a few seconds (allow 15 s), saves no files and prints:

1. both networks' parameter counts and the input batch: observations `[64, 452]`, action masks
   `[64, 26]`;
2. **forward pass** of the trained set network: logits `[64, 26]`, values `[64]`, and three
   example decisions next to Slack-aware's;
3. **backward pass** of a fresh set network: policy and value losses, gradient norms, and the loss
   before and after one Adam step;
4. a whole day for each trained network and for Slack-aware (a single day; it says little about
   average performance).

## Replays (`assets/`, nothing to install)

| File | What it shows |
|---|---|
| [replay_set_network.html](assets/replay_set_network.html) | The trained set network's whole day: the situation at each decision and up to three top choices with their probabilities. |
| [replay_slack.html](assets/replay_slack.html) | Slack-aware on the same day, for comparison. |
| [set_network_backlog.mp4](assets/set_network_backlog.mp4) | The set network's replay as a video, over the busy part of the day (09:58–16:38, 5 min 18 s). |
| [demo.png](assets/demo.png) | The decision pictured above. |

All show the same unseen medium-load day (demand seed 100000). Open an HTML file in a browser; it
works offline. It opens at 09:58, where orders start to pile up; **Start of the day** goes back to
08:00.

- **Play** (or Space) plays and pauses.
- **1×**, **2×**, **3×** set the speed. At 1× a flight plays at one simulated minute per second;
  box handling, charging and waiting play faster.
- **Previous / Next decision** (← →) step through the decisions; tick **Stop at each decision** to
  pause at every decision while playing. Click or drag the timeline to jump.
- **How to read this screen** explains the display.

At 1× the clock stops for about 2 s at each decision to show the choices. **This pause is for
reading, not the network's decision time:** the network decides in milliseconds.

## Other commands

Run from the unzipped folder with the environment active. Times are rough CPU estimates.

```bash
python scripts/demo.py --policy set --seed 100000 --video   # replay, picture, video (~10 min)
python scripts/train_bc.py                                  # retrain imitating Slack-aware (~6 min)
python scripts/train_bc.py --teacher rollout                # retrain imitating rollout (~11 min)
python scripts/evaluate.py --checkpoints checkpoints/bc_mlp.pt checkpoints/bc_set.pt   # ~1 min
python scripts/evaluate.py --methods slack,rollout --out outputs/evaluation_rollout.csv  # ~5 min
```

- `demo.py` saves the replay and picture in `outputs/` within seconds; `--video` adds the video.
  It is also the environment's usage example (`--policy slack` runs the teacher).
- `train_bc.py` overwrites `bc_mlp.pt`, `bc_set.pt` and `bc_history.json` in `checkpoints/` (with
  `--teacher rollout`, the `bc_rollout_*` files).
- `evaluate.py` runs the given policies on the same 40 days per load, prints a summary and saves a
  CSV (`outputs/evaluation.csv` by default). Without `--methods` it runs Random, EDF, FCFS,
  nearest-first and Slack-aware.

## Environment and networks

| Environment | |
|---|---|
| Observation | `Box(452)`: 80 global features, 12 waiting-order slots × 27, 3 cargo slots × 16 |
| Action | `Discrete(26)`: 12 pick up, 6 deliver, 6 charge, return, wait; allowed ones in `info["action_mask"]` |
| Reward | +1 on time, +0.1 late, −0.1 expired, −0.001 per battery % used |
| Episode | one day at low, medium or high demand; `reset(options={"demand_seed": ..., "load": ...})` |

Both networks take observations `[B, 452]` and masks `[B, 26]` and return action logits `[B, 26]`
(forbidden actions set to −10⁹) and values `[B]`; a network policy takes the allowed action with
the highest logit.

| Network | Structure | Parameters |
|---|---|---:|
| MLP (control) | separate actor and critic, each 452 → 256 → 256, tanh | 370,459 |
| Set/attention | shared order and cargo encoders (width 64), four-head attention, shared pick-up scorer | 62,928 |

## Main files

```
medical_uav/agents/     networks.py (both networks), policy.py (inference, checkpoints), bc.py (imitation)
medical_uav/env/        the environment; replay.py, replay_player.html, video.py (replays)
medical_uav/baselines/  Random, EDF, FCFS, nearest-first, Slack-aware, rollout planner
medical_uav/            evaluation.py, metrics.py, splits.py (which days are for what)
scripts/                demo_network.py, demo.py, train_bc.py, evaluate.py
checkpoints/            bc_set.pt, bc_mlp.pt, bc_rollout_set.pt, bc_rollout_mlp.pt, *history.json
assets/                 replays, video, picture
environment.yml         the exact environment
```
