# Dynamic Medical Pickup-and-Delivery with a Single UAV

NUS robotics RL course project, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang).

A single drone serves a medical center and five hospitals for one operating day. Orders
(plasma, medicine, lab samples, inter-hospital supplies) arrive at random times with two daily
peaks; each type has its own weight, service time limit and expiry. The drone carries up to
3 boxes / 2 kg, and its battery drains faster with heavier loads, so it must decide what to pick
up, where to deliver, and when and where to recharge. This repository contains the Gymnasium
environment (milestone 1); learning agents follow in later milestones.

Design document: `docs/superpowers/specs/2026-10-10-medical-uav-design.md`.

## Installation

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
pip install -e .
```

Tested on Linux with Python 3.11. If conda stops with a Terms-of-Service message about the
Anaconda `defaults` channel, either accept those terms (`conda tos accept ...`) or use conda-forge
only:

```bash
conda create -n medical-uav-rl --override-channels -c conda-forge python=3.11 pip
conda activate medical-uav-rl
pip install -r requirements.txt && pip install -e .
```

## Quick start

```bash
# one day with the EDF baseline, prints metrics and saves outputs/demo_edf_medium_0.gif
python scripts/demo.py --policy edf --load medium --seed 0

# random valid actions on a low-load day
python scripts/demo.py --policy random --load low --seed 1

# metrics only / PNG snapshots at chosen minutes
python scripts/demo.py --policy edf --no-video --snapshots 0 100 160

# calibration table (load levels, map statistics, throughput)
python scripts/calibrate.py --map-check 200 --benchmark
```

Using the environment directly:

```python
import numpy as np
from medical_uav.env import MedicalUAVEnv

env = MedicalUAVEnv(load="medium", render_mode="rgb_array")
obs, info = env.reset(seed=0)
done = False
while not done:
    action = np.random.choice(np.flatnonzero(info["action_mask"]))
    obs, reward, done, truncated, info = env.step(action)
frame = env.render()
```

## Repository structure

```
medical_uav/
  env/
    config.py        all parameters (map, drone, orders, rewards)
    physics.py       flight time / energy, charging time, delivery reward
    astar.py         8-connected grid A*
    map_gen.py       random map with no-fly zones, A* distance matrix, validity check
    orders.py        Order dataclass and the seeded daily order stream
    simulator.py     event-driven simulation of one day (pick, deliver, charge, return, wait)
    actions.py       the 26-action table
    masks.py         action mask (capacity, expiry and battery safety rules)
    observation.py   415-dim observation, pending-slot ranking
    uav_env.py       MedicalUAVEnv (Gymnasium wrapper)
    render.py        matplotlib rendering replayed from the simulation history
  baselines/
    heuristics.py    random and EDF policies, shared charging rule
  metrics.py         per-episode metrics
scripts/
  demo.py            run an episode, print metrics, save an animation
  calibrate.py       load calibration, map statistics, throughput
tests/               pytest suite (86 tests)
docs/                design spec, implementation plan, report material
```

## Environment summary

- **Map:** 16 x 16 grid (4 km x 4 km) with 2–4 rectangular no-fly zones; nodes M (center, also
  charger C1), H1–H5, and a standalone charger C2. A* routes between all nodes are computed once.
  The main experiments use `map_seed = 0`.
- **Episode:** orders are released during 480 minutes (two peaks), then the day continues until every
  order is delivered, expired or rejected. Base rates: low 0.04, medium 0.06, high 0.10 orders/min.
- **Actions (`Discrete(26)`) with a mask:**

  | index | action |
  |---|---|
  | 0–11 | `PICK(k)`: fly to pending slot k's origin and load it |
  | 12–17 | `DELIVER(n)`: fly to facility n and unload all boxes for n |
  | 18–23 | `CHARGE(station, target)`: C1/C2 × 60/80/100 % |
  | 24 | `RETURN_TO_CENTER` |
  | 25 | `WAIT` until the next order release/expiry (max 10 min) |

  The mask (`info["action_mask"]` or `env.action_masks()`) removes actions that break capacity,
  cannot load before expiry, or would leave too little battery to reach a charger with a 15 %
  reserve. A masked action passed to `step` is executed as WAIT.
- **Observation (`Box(415)`):** global state (time, demand profile, battery, load, location, travel
  times), the 12 most relevant pending orders (on-time-feasible first), and the 3 onboard boxes.
- **Reward:** +1 per on-time delivery, decreasing linearly to −0.5 at expiry for late deliveries,
  −0.5 per expired or rejected order, −0.001 per % of battery used.
- **`reset` options:** `demand_seed`, `load` (`"low"`/`"medium"`/`"high"`) or `lam` (float).
- **`info` keys:** `action_mask`, `t`, `elapsed`, `invalid_action`, `action_name`, `events`.

## Tests

```bash
python -m pytest -q
```

## Libraries

numpy (numerics), gymnasium (environment API and `check_env`), matplotlib (rendering), imageio +
imageio-ffmpeg (gif/mp4 export), pytest (tests). The environment, A*, map generator, simulator,
masks, observation, renderer and baselines are written by the group; no external environment or
RL code is used.

## Use of AI tools

This project was developed with the help of AI assistants (Anthropic Claude and OpenAI Codex) for
design discussion, code writing, code review and documentation. The group reviewed, tested and is
responsible for all submitted work.
