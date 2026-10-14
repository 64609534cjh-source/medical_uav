# Dynamic Medical Pickup-and-Delivery with a Single UAV

NUS ME5418 robotics reinforcement-learning project, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang).
Milestone 1: the Gymnasium environment and rule-based baseline policies.

One delivery drone serves a medical center and five hospitals for one simulated day. Orders of four
types (plasma, medicine, lab samples, inter-hospital supplies) appear at random times, with two
daily peaks. Each type has its own weight, on-time limit and expiry time. The drone carries at most
3 boxes and 2 kg, its battery drains faster with heavier cargo, and it must never run out of
charge. At each decision it chooses what to pick up, where to deliver, when and where to recharge,
or to wait.

![A simulated day at 11:42](assets/demo.png)

*Slack-aware baseline, medium load, demand seed 0. Left: the 4 km x 4 km map with no-fly zones
(red), the medical center M (also charger C1), hospitals H1–H5, charger C2, and the drone (+)
flying to H5 with three boxes. Right: clock, battery with the 15 % reserve mark, cargo with minutes
left to each deadline, and the pending orders sorted by deadline.*

## Install and run

Run all commands from the repository root. Tested on Linux with Python 3.11.

```bash
conda create -n medical-uav-rl --override-channels -c conda-forge python=3.11 pip
conda activate medical-uav-rl
pip install -r requirements.txt
pip install -e .
```

Alternatively, `conda env create -f environment.yml` installs the same pinned versions; then run
`conda activate medical-uav-rl` and `pip install -e .`.

**1. Walk through the API** (about 50 readable lines):

```bash
python scripts/example.py
```

It creates the environment, prints the drone's state, the visible orders and the allowed actions,
takes five decisions while printing their duration, reward and events, saves the current frame to
`outputs/example_frame.png`, and finally runs a whole day:

```
t = 1.0 min | at M | battery 100.0 % | 0 box(es) on board
  slot 0: supplies H5->H1  0.75 kg  deadline in 75 min
  allowed: PICK(0), CHARGE(C2,100), WAIT

> PICK supplies H5->H1
  took 3.9 min, reward -0.006, events: pickup #0 at 4.8
...
full day: 64 orders, on time 70%, expired 19%, return 40.39
```

**2. Watch a day:**

```bash
python scripts/demo.py --live   # play the day in a window; without a display it saves a GIF
python scripts/demo.py          # save outputs/demo_slack_medium_0.gif
```

**3. Run the tests** (183 tests, including Gymnasium's `check_env`):

```bash
python -m pytest -q
```

## Using the environment in code

```python
import imageio.v2 as imageio
import numpy as np
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv

env = MedicalUAVEnv(load="medium", render_mode="rgb_array")
obs, info = env.reset(options={"demand_seed": 0})   # the same seed always gives the same day
print(obs.shape, info["t"])                          # (415,) 0.0

allowed = np.flatnonzero(info["action_mask"])        # actions allowed right now
print([env.describe(int(a)) for a in allowed])

action = slack_aware_policy(env)                     # a baseline picks one allowed action
print("doing", env.describe(action))                 # describe before stepping: slots change
obs, reward, terminated, truncated, info = env.step(action)
print(reward, info["elapsed"], info["events"])       # reward, minutes taken, what happened

imageio.imwrite("frame.png", env.render())           # picture of the current state
```

- `reset(options=...)` accepts `demand_seed` (which day of orders), `load` (`"low"`, `"medium"`,
  `"high"`) or `lam` (a base arrival rate in orders per minute). `reset(seed=s)` without a
  `demand_seed` draws a random training day.
- `info` holds `action_mask`, `t` (simulated minutes), `elapsed`, `invalid_action`, `action_name`
  and `events` (releases, pickups, deliveries, expiries during the step).
- An episode is one day; it ends when no more orders can arrive and every order is delivered,
  expired or rejected (about 100 decisions). `truncated` is always `False`.
- To run a baseline for a whole day and get its metrics:
  `run_episode(env, slack_aware_policy, demand_seed=0)` from `medical_uav.baselines.heuristics`.

## How the environment works

**One step = one complete action.** The clock jumps forward by the action's real duration (flying,
2 minutes per box loaded or unloaded, charging, waiting). Orders that appear or expire during the
action are handled at their exact times.

**Map.** A 16 x 16 grid of 0.25 km cells with 2–4 rectangular no-fly zones and 7 nodes. Flight
routes between nodes are shortest paths around the no-fly zones (A* search), computed once. The
main experiments use one fixed map (`map_seed = 0`).

**Orders.** The day starts at 08:00. Orders arrive at random (a Poisson process) until 16:00. The
base arrival rate is multiplied by 2.0 during the peaks 09:00–11:00 and 13:00–14:30 and by 0.6
otherwise.

| Type | Route | Weight (kg) | On time within | Expires after | Share |
|---|---|---|---|---|---|
| plasma | center → hospital | 0.6–0.9 | 30 min | 60 min | 20 % |
| medicine | center → hospital | 0.3–0.6 | 90 min | 180 min | 30 % |
| lab sample | hospital → center | 0.2–0.4 | 60 min | 120 min | 30 % |
| supplies | hospital → hospital | 0.4–0.8 | 75 min | 150 min | 20 % |

**Drone.** Flies 1 km per minute. A flight uses `3.5 × distance_km × (1 + 0.25 × cargo_kg)` % of
the battery; charging adds 5 % per minute at M (C1) or C2. At most 30 orders can wait; further
arrivals are rejected. A box that expires on board stays there, still counting toward the 3-box and
2 kg limits, until the drone next stops at M or a hospital, where unloading it takes 2 minutes.

**Actions** (26, with a mask of allowed actions):

| Index | Action | Meaning |
|---|---|---|
| 0–11 | `PICK(k)` | fly to the order shown in pending slot k and load it |
| 12–17 | `DELIVER(f)` | fly to facility f (M, H1–H5) and unload every box for it |
| 18–23 | `CHARGE(s, x)` | fly to charger C1 or C2 and charge to 60, 80 or 100 % |
| 24 | `RETURN` | fly back to the medical center |
| 25 | `WAIT` | wait until the next order arrives or expires, at most 10 minutes |

The mask (`info["action_mask"]` or `env.action_masks()`) allows an action only if the cargo fits (3
boxes, 2 kg), the order can be loaded before it expires, and afterwards the drone can still reach a
charger while keeping 15 % in reserve. So the drone can never be stranded. A forbidden action
passed to `step` is executed as `WAIT`.

**Observation** (415 numbers, normalized and clipped to [−10, 10]): 31 about the day and the
drone (time, demand now and ahead, battery, cargo, location, travel times to every node); 12 slots
of 28 numbers for the most relevant waiting orders (those that can still be on time come first);
3 slots of 16 numbers for the boxes on board. Unused slots are zero.

**Reward.** +1 for each box delivered on time; a late box earns less the later it is, down to −0.5
at its expiry; −0.5 for each expired or rejected order; −0.001 per % of battery used.

## Baselines and checks

| Policy | Rule |
|---|---|
| Random | a random allowed action |
| EDF (earliest deadline first) | serve the pickup or delivery with the earliest deadline |
| Slack-aware | unload at the current node first; then pick up an order here whose destination is already on board; then charge if due; then earliest deadline first, with orders that can no longer be on time moved last |

Charging is due below 20 % battery, or below 80 % when nothing can be picked up or delivered. EDF
checks it before anything else; Slack-aware checks it after its two rules for the current node.

On-time rate of a day = on-time deliveries / all orders released that day (rejected ones
included). Mean over 40 development days per load:

| Load | Base orders per minute | Orders per day | Slack-aware | EDF |
|---|---|---|---|---|
| low | 0.04 | about 22 | 1.00 | 1.00 |
| medium | 0.08 | about 47 | 0.84 | 0.51 |
| high | 0.12 | about 68 | 0.65 | 0.26 |

The environment runs about 9,000 decisions per second on one CPU core. To reproduce the table, the
charging-threshold tuning, the map statistics and the speed measurement:

```bash
python scripts/calibrate.py --tune-theta --map-check 200 --benchmark
```

More demo options: `--policy {slack,edf,random}`, `--load {low,medium,high}`, `--seed N` (demand
seed), `--no-video`, and `--snapshots 0 100 160` (PNG frames at those minutes, saved in
`outputs/`).

## Code guide

```
medical_uav/
  env/
    config.py        all settings: map, drone, order types, demand, rewards (EnvConfig)
    physics.py       flight time, flight energy, charging time, delivery reward
    astar.py         shortest path on the grid around no-fly zones
    map_gen.py       random map: no-fly zones, node positions, routes, validity check
    orders.py        orders and the seeded order stream of one day
    simulator.py     event-driven simulation of the day: the five actions, releases, expiries
    actions.py       the 26 actions and their indices
    masks.py         which actions are allowed (capacity, expiry and battery rules)
    observation.py   the 415-number observation and which orders fill the 12 slots
    uav_env.py       MedicalUAVEnv, the Gymnasium environment
    render.py        drawing frames and saving animations
  baselines/
    heuristics.py    Random, EDF and Slack-aware policies; run_episode
  metrics.py         per-day metrics: on-time rate overall and per type, energy, charging time
  splits.py          which demand seeds are used for training, development and testing
scripts/
  example.py         API walkthrough
  demo.py            watch or record a day
  calibrate.py       load calibration, map statistics, speed
tests/               pytest suite
```

The simulator is plain Python with no Gymnasium code; `uav_env.py` wraps it. The baselines, the
renderer and the metrics read the same simulator, so they always agree on time, battery and orders.
To change a setting, pass an `EnvConfig` to `MedicalUAVEnv(config=...)`.

## Libraries

numpy (arrays, random numbers), gymnasium (environment interface, `check_env`), matplotlib
(drawing), imageio and imageio-ffmpeg (GIF and MP4 files), pytest (tests). The environment,
routing, simulator, baselines and renderer are written for this project; no external environment or
RL code is used.