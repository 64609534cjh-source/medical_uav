# Medical UAV Delivery: Environment, Networks and a PPO Agent

A Gymnasium environment in which one drone picks up and delivers medical orders between a medical
center and five hospitals over one simulated day. Orders arrive at random and have deadlines; the
drone must avoid no-fly zones, carry at most 3 boxes and 2 kg, and never run out of battery.
The package contains rule-based baselines, a planner that looks one step ahead, two
policy/value networks (an MLP and a set/attention network) and a dispatcher trained with our
own masked PPO implementation.

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). Milestone 3: the learning agent.

![A decision of the PPO agent](assets/demo.png)

## See it first

Download `assets/replay_ppo.html` and open it in a web browser (double-click; no installation or
internet connection is needed). It replays one day, medium load and demand seed 100000 (a day
outside training), driven by the PPO agent: play, pause, step from decision to decision or drag
the timeline. Each decision is shown with the situation it was taken in and the agent's three
most likely actions.

| File in `assets/` | What it shows |
|---|---|
| `replay_ppo.html` | the PPO agent, one whole day, in a web browser |
| `replay_slack.html` | the Slack-aware rule, the strongest hand-written rule, on the same day |
| `replay_set_network.html` | the set network trained by imitation (Milestone 2) on the same day |
| `ppo_explained.mp4` | eight decisions of the PPO agent, each held before, during and after its action |
| `ppo_day.mp4` | the whole day in under five minutes at smart speed: flights at 1 and everything else at 8 simulated minutes per second, a short pause at each decision |
| `set_network_explained.mp4`, `set_network_day.mp4` | the same for the set network trained by imitation |
| `demo.png` | the picture above |

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

## Run the trained agent

```bash
python scripts/demo.py --policy ppo
```

The PPO agent in `checkpoints/ppo_set.pt` dispatches one day (medium load, demand seed 0) and
prints the results, which start with:

```
Completed at 16:37 (t=517.0 min) after 126 decisions
  Orders   70 released: delivered 61/70 (87.1%), expired 9/70 (12.9%)
  On time  58/70 (82.9%); late 3 (mean 10.5 min late)
```

On the same day Slack-aware, the strongest rule, delivers 53 of the 70 orders, 47 on time.
The replay page and a frame are saved in `outputs/` (add `--video` for the two videos), as
described in [Run the day demo](#run-the-day-demo).

## Train the agent

```bash
python scripts/train_ppo.py --net set --seed 11 --decisions 6e6 --out runs/set_s11
```

Masked PPO on randomly drawn training days at all three loads (about an hour on 4 CPU threads).
Every 0.5 M decisions it saves a checkpoint and the greedy on-time rate on 20 selection days;
`updates.csv` logs the losses, entropy, KL and clip fraction of every update. Options:
`--net mlp`, `--reward m1` or `--reward ontime` (reward variants), `--init checkpoints/bc_set.pt`
(start from the imitation-trained network after a value warm-up).

```bash
python scripts/select_checkpoints.py runs/set_s11 --out runs/selection.json
python scripts/plot_training.py "PPO:runs/set_s11" --out outputs/training_curves.png
```

The first chooses one checkpoint per run by a rule fixed before training (best medium/high
on-time rate on 100 selection days, keeping the low load at least 97 % on time and every order
type within 3 points of Slack-aware); the second draws the learning curves.

## Test the trained agent

```bash
python scripts/evaluate.py --methods slack,rollout --checkpoints checkpoints/ppo_set.pt --processes 8
```

It runs every policy on the same 40 days at each load, writes one row per day to
`outputs/evaluation.csv` and prints each method against Slack-aware with a 95 % bootstrap
interval over days. `python scripts/analyze_behaviour.py --checkpoints checkpoints/ppo_set.pt`
compares how the policies serve those days (boxes per stop, charging, waiting, peak hours).

On-time rate (orders delivered within their time limit) on the same 40 days per load
(`--seeds screen`, demand seeds 100000–100039):

| Policy | Low | Medium | High |
|---|---|---|---|
| Random | 55.5% | 26.3% | 22.6% |
| EDF (earliest deadline first) | 99.6% | 40.6% | 27.5% |
| FCFS (first come, first served) | 97.0% | 49.3% | 32.1% |
| Nearest-first | 99.2% | 71.6% | 53.2% |
| Slack-aware (strongest rule) | 99.3% | 81.0% | 65.9% |
| One-step lookahead planner (`rollout`) | 99.9% | 89.7% | 76.8% |
| Set network trained by imitation (`bc_set.pt`) | 99.5% | 79.5% | 64.4% |
| **PPO agent (`ppo_set.pt`)** | **100.0%** | **92.3%** | **84.2%** |
| PPO, mean (range) of 5 training seeds | 100.0% | 91.7% (90.8–92.3) | 82.9% (81.9–84.2) |

The PPO agent was chosen among the five training seeds on 100 other days (`--seeds select`);
the last row shows how much the result varies with the training seed. A decision takes the agent
about 0.5 ms and the planner about 30 ms on one CPU thread.

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

It then saves, within a few seconds, in `outputs/`:

- `demo_slack_medium_seed0_replay.html`, the whole day in a web browser, as in
  [See it first](#see-it-first)
- `demo_slack_medium_seed0_decision.png`, one decision worth a look

With `--video` it also saves (10–15 minutes):

- `demo_slack_medium_seed0_explained.mp4`, eight consecutive decisions, each shown before,
  during and after its action
- `demo_slack_medium_seed0_day.mp4`, the whole day at smart speed

| Option | Values |
|---|---|
| `--policy` | `slack` (default), `edf`, `random`, `mlp`, `set` (the networks trained by imitation), `ppo` (the PPO agent); for the networks the top choices are shown |
| `--load` | `low`, `medium` (default), `high` |
| `--seed` | which day of orders (default `0`) |
| `--video`, `--explain-count`, `--fps` | also save the two videos; decisions in the explanation video (8); video frames per second (12) |

`scripts/demo.py` is also the usage example: it creates the environment, starts a day, lets a
policy choose actions until the day is over, and then shows the day.

## The environment

| | |
|---|---|
| Observation | `Box(452)`: the drone, travel times between all nodes, up to 12 waiting orders (those that can still be on time first) and the cargo on board |
| Action | `Discrete(26)`: pick up, deliver, charge, return to the center, wait; `info["action_mask"]` marks the allowed ones |
| Reward | +1 per on-time delivery, +0.1 per late delivery, −0.1 per expired order, −0.001 per battery % used |
| Episode | one day, about 80–120 decisions |
| `reset` options | `demand_seed` (which day of orders), `load` (`"low"`, `"medium"` or `"high"`) |
| Baselines | `random_policy`, `edf_policy`, `fcfs_policy`, `nearest_policy`, `slack_aware_policy` in `medical_uav.baselines.heuristics`; `rollout_policy` (one-step lookahead over Slack-aware, visible orders only) in `medical_uav.baselines.rollout` |

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

The rules of the environment are described in the Milestone 1 report, the networks in the
Milestone 2 report and the PPO agent in the two-page report `ME5418_Group17_M3_Report.pdf`.

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
    render.py        drawing a day for env.render()
    replay.py        a finished day as a record; the replay web page
    replay_player.html  the replay web page (the record is put inside it)
    video.py         frames and videos of a finished day, in the replay's style
  agents/
    networks.py      the MLP and the set/attention network
    policy.py        a network as a policy; saving and loading checkpoints
    bc.py            behaviour cloning: recording a teacher, training a network
    ppo.py           masked PPO: collection, advantages, update, value warm-up
  baselines/
    heuristics.py    Random, EDF, FCFS, nearest-first and Slack-aware policies
    rollout.py       the one-step lookahead planner
  evaluation.py      running policies on the same days and comparing them
  metrics.py         per-day metrics
  splits.py          seeds for training, screening, model selection and testing
scripts/
  demo_network.py    network demo: forward and backward pass, a day with each network
  demo.py            run and show one day; also the usage example
  train_bc.py        behaviour cloning of Slack-aware for both networks
  train_ppo.py       PPO training of one run
  select_checkpoints.py  choose one checkpoint per run on the selection days
  plot_training.py   learning curves of groups of runs
  evaluate.py        compare policies on the same days
  analyze_behaviour.py   how policies serve the same days
checkpoints/
  ppo_set.pt         the PPO agent (set network); ppo_set.json, how it was trained and chosen
  bc_mlp.pt, bc_set.pt  the imitation-trained networks; bc_history.json, their training curves
assets/
  *.html, *.mp4      the replays and videos listed in "See it first"
  demo.png           the picture at the top of this page
ME5418_Group17_M3_Report.pdf   the two-page report
environment.yml      conda environment
requirements.txt     the same packages for pip
pyproject.toml       package definition for `pip install -e .`
```
