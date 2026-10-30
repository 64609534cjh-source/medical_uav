# Medical UAV Delivery: A PPO Learning Agent

NUS ME5418, Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang). **Milestone 3: the learning agent.**

One drone collects and delivers medical orders between a medical center and five hospitals in
our Gymnasium environment. It must meet deadlines, avoid no-fly zones, respect cargo limits and
manage its battery. A set/attention network chooses actions using masked proximal policy
optimization (PPO), implemented in `medical_uav/agents/ppo.py`.

The final agent was trained from scratch on 500 random maps (map seeds 1000–1499), with reward
variant `v2_expiry05`, for 12 million decisions per run and five training seeds.
Design and results are in the two-page report `ME5418_Group17_M3_Report.pdf`.

![A decision in the PPO replay](assets/demo.png)

## Install

Requires Conda; no GPU. Tested on Linux. `environment.yml` pins the environment, including
Python 3.11.17 and PyTorch 2.14.1 CPU. Open a terminal in the unzipped folder:

```bash
conda env create -f environment.yml
conda activate medical-uav-rl
python -m pip install -e .
```

If `medical-uav-rl` already exists, add `-n medical-uav-m3` to the first command and activate
`medical-uav-m3` instead. Run all commands below from this folder with the environment active.
Times are approximate, measured on a 24-core Linux CPU machine.

## Run the shipped agent

```bash
python scripts/demo.py --policy ppo --load high --seed 100007
python scripts/demo.py --policy slack --load high --seed 100007
python scripts/demo.py --policy ppo --load high --seed 100007 --map 2000
```

The first command takes about 2 s, prints the day's results and saves
`outputs/demo_ppo_high_seed100007_replay.html` and
`outputs/demo_ppo_high_seed100007_decision.png`.
The second runs Slack-aware on the same arrivals and map. The third uses an unseen test map;
its output names include `_map2000`.

Options: `--policy random|edf|slack|set|mlp|ppo`, `--load low|medium|high`, `--seed` for the demand
day, `--map` for the geography, and `--checkpoint PATH` for a compatible saved network with
`ppo`, `set` or `mlp`. The `set` and `mlp` defaults load the M2 imitation-trained networks.
Add `--video` to save a `_backlog.mp4` video; rendering takes a few minutes.

The shipped agent is `checkpoints/ppo_set.pt`: training seed 11, checkpoint at 10,500,096
decisions. `checkpoints/ppo_set.json` records its hash, training command and selection rule.
The training seed was fixed before the final test.

## Replays without installation

Open an HTML file in `assets/` in a browser; it works offline.

| File | What it shows |
|---|---|
| [replay_ppo.html](assets/replay_ppo.html) | The shipped agent's whole day; at each decision, the situation and its top choices with their probabilities. |
| [replay_slack.html](assets/replay_slack.html) | Slack-aware on the same day. |
| [demo.png](assets/demo.png) | The decision pictured above. |

Both show high load, demand seed 100007, map 0: the development day on which PPO's lead over
Slack-aware is closest to its median over the 40 development days (PPO 78.8 %, Slack-aware
62.1 % on time). They are the outputs of the first two commands above.
**Play** or Space plays and pauses; ← and → step through decisions.
Drag the timeline to jump, or tick **Stop at each decision**.
**How to read this screen** explains the display.
Pauses at decisions allow time to read the choices; they are not computation time.

## A short training run

```bash
python scripts/train_ppo.py --net set --seed 11 --maps train --reward v2_expiry05 --decisions 5e5 --eval-every 1e5 --eval-days 10 --threads 2 --out outputs/quick_s11
```

This takes about 5 min. It trains on randomly drawn days at all three loads and prints greedy
on-time rates on 10 selection days at medium and high load, rotating over selection maps.
In the checked run, medium/high rates were 43.4/28.7% before training, 80.7/67.2% at about
100,000 decisions and 85.1/69.1% at about 500,000 decisions.

It saves `ckpt_*.pt`, `evals.csv`, `updates.csv` and `run.json` in `outputs/quick_s11/`.
Updates contain 2,048 decisions, so evaluation and checkpoint counts round up to whole updates.
Plot its learning curve, with Slack-aware on the same days and maps (about 30 s):

```bash
python scripts/plot_training.py "Quick PPO:outputs/quick_s11" --eval-days 10 --out outputs/quick_learning.png
```

Run the final checkpoint from this short run with:

```bash
python scripts/demo.py --policy ppo --checkpoint outputs/quick_s11/ckpt_00501760.pt --load high --seed 100007
```

This uses the same output names as the shipped-agent demo and replaces those replay files.
The short run demonstrates learning; it does not reproduce the final agent.

## A quick evaluation

```bash
python scripts/evaluate.py --methods slack,insertion,rollout --checkpoints checkpoints/ppo_set.pt --loads medium,high --days 5 --out outputs/quick_eval.csv
```

About 25 s. Each policy runs the same five development (screen) days per load on map 0.
The script saves one CSV row per policy-day and prints on-time and served rates, plus paired
differences from Slack-aware with 95% bootstrap intervals. This is a functional check;
the final results below use separate test days.

## Final results

Mean daily on-time rate (%). PPO is the mean of five independently trained seeds.
Map 0 uses 300 new demand days per load; unseen maps use 30 maps × 10 days per load.

| Policy | Map 0 medium | Map 0 high | Unseen medium | Unseen high |
|---|---:|---:|---:|---:|
| Slack-aware | 79.6 | 63.3 | 82.6 | 66.1 |
| Insertion | 85.7 | 69.1 | 87.8 | 71.3 |
| Visible-order rollout | 89.0 | 74.6 | 90.5 | 76.1 |
| Scenario rollout (16 futures) | 89.2 | 76.5 | 90.4 | 77.8 |
| Rolling horizon | 92.2 | 79.4 | 93.2 | 80.6 |
| PPO | 91.8 | 80.8 | 92.7 | 81.7 |

Serial decisions take about 0.23 ms for PPO, compared with 564/1,373 ms for rolling horizon at
medium/high load. See the report for comparisons and uncertainty, `evidence/final_test/` for
the complete results, and `evidence/latency/` for serial timing samples.

## Full reproduction (optional)

Train one final-agent run, about 3–4 h:

```bash
python scripts/train_ppo.py --net set --seed 11 --maps train --reward v2_expiry05 --decisions 12e6 --threads 2 --out runs/f12_mm_e05_s11
```

The five final runs used seeds 11, 22, 33, 44 and 55, each with its own output directory.
The final agent uses checkpoint selection, rather than automatically taking the last checkpoint:

```bash
python scripts/select_checkpoints.py runs/f12_mm_e05_s11 --maps select --days 20 --processes 8 --out runs/f12_selection.json
```

Selection evaluates the first 20 selection days on each of maps 100–109. Eligible checkpoints
must reach 97% on time at low load and stay within 3 percentage points of Slack-aware for every
order type at medium and high load. Among them, the highest medium/high mean wins.

To rerun the registered final test with the supplied frozen checkpoints:

```bash
python scripts/locked_test.py run --days test --processes 8 --out outputs/final_rerun
python scripts/locked_report.py outputs/final_rerun > outputs/final_rerun/report.txt
```

This runs 109,200 policy-days: about 11 h with 8–24 workers here, much longer on a laptop.
The rolling-horizon planner dominates the time. Git must be available on PATH; use unchanged
code and the supplied checkpoints. Interrupted runs resume from saved days.
Do not run the `freeze` command: the entrants and their hashes are already supplied.

For a short serial decision-time benchmark:

```bash
python scripts/latency.py --methods slack,horizon --checkpoints checkpoints/ppo_set.pt --loads medium,high --decisions 100 --passes 1 --out outputs/latency.csv
```

This times policies on one CPU thread, separately from parallel evaluation; horizon is slow.
The short sample is a timing check. The published samples are in `evidence/latency/`.

## M2 network demo and imitation training

```bash
python scripts/demo_network.py
python scripts/train_bc.py
python scripts/train_bc.py --teacher rollout
```

The network demo takes a few seconds: parameter counts, forward and backward passes on real
observations, then a day with each trained network and Slack-aware. It saves no files.
Behaviour cloning (BC) trains both networks to imitate a teacher. The two training commands
overwrite the corresponding `bc_*` or `bc_rollout_*` checkpoints and histories.

## Environment and networks

| Environment | |
|---|---|
| Observation | `Box(452)`: 80 global features, 12 waiting-order slots × 27, 3 cargo slots × 16 |
| Action | `Discrete(26)`: 12 pick up, 6 deliver, 6 charge, return, wait; allowed actions in `info["action_mask"]` |
| PPO training reward | `v2_expiry05`: +1 on time, +0.1 late, −0.5 expired, −0.001 per battery % used |
| Episode | one day at low, medium or high demand; `reset(options={"demand_seed": ..., "load": ...})` |

The default environment and day demo retain the `v2` expiry penalty of −0.1; the training
commands above explicitly select −0.5.

Both networks take observations `[B, 452]` and masks `[B, 26]`, returning action logits `[B, 26]`
and values `[B]`. Forbidden logits are set to −10⁹. Inference chooses the highest allowed logit.

| Network | Structure | Parameters |
|---|---|---:|
| Multilayer perceptron (MLP) | separate actor and critic, each 452 → 256 → 256, tanh | 370,459 |
| Set/attention | shared order and cargo encoders, width 64, four-head attention, shared pick-up scorer | 62,928 |

Baselines include Random, earliest deadline first (EDF), first come first served (FCFS),
nearest-first, Slack-aware, insertion, visible-order rollout, scenario rollout with 16 sampled
futures, rolling horizon and energy-management variants.

## Main files

```text
medical_uav/agents/     ppo.py (masked PPO), networks.py (both networks),
                        policy.py (inference and checkpoints), bc.py (imitation)
medical_uav/env/        Gymnasium environment, simulator, maps, masks and observations;
                        replay.py, replay_player.html, video.py (replays)
medical_uav/baselines/  heuristics.py, insertion.py, rollout.py, scenario.py,
                        horizon.py, energy.py
medical_uav/            evaluation.py, metrics.py, splits.py (day and map seed ranges)
scripts/
  demo.py               run one day; save replay, picture and optional backlog video
  demo_network.py       M2 forward/backward passes and whole-day network demo
  train_ppo.py          train one PPO run; save checkpoints, evaluations and updates
  train_bc.py           imitate Slack-aware or rollout with both networks
  evaluate.py           paired development evaluation and per-day CSV
  select_checkpoints.py choose one eligible checkpoint per run on selection days
  plot_training.py      learning curves and Slack-aware comparison
  locked_test.py        run the frozen final-test entrants and check hashes
  locked_report.py      final-test statistics, CSVs and summary JSON
  latency.py            serial decision-time benchmark
checkpoints/            ppo_set.pt and ppo_set.json (shipped agent and provenance);
                        entrants/ (41 frozen PPO checkpoints and manifest.json);
                        bc_set.pt, bc_mlp.pt (Slack-aware imitation);
                        bc_rollout_set.pt, bc_rollout_mlp.pt (rollout imitation);
                        bc_history.json, bc_rollout_history.json
evidence/final_test/    report.txt, map0.csv, maps.csv, summary.json
evidence/selection/     f12_select_*.json (checkpoint selection)
evidence/latency/       serial timing CSVs
evidence/training/      evals.csv for all final-test PPO runs, with run logs
evidence/registration/  2026-10-29-final-test-preregistration.md
assets/                 offline replays and picture
ME5418_Group17_M3_Report.pdf  the two-page report
environment.yml         exact Conda environment
pyproject.toml          package definition for editable installation
```
