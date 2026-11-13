# Project summary (Group 17): experiments and results

A summary of the whole project: the problem, what each milestone did, the final method, the baselines, the final
test and its results, and where the code, the trained models and the result files are. The three submitted
milestone reports are next to this file: `ME5418_Group17_M1_Report.pdf`, `ME5418_Group17_M2_Report.pdf`,
`ME5418_Group17_M3_Report.pdf`.

## 1. The problem

One drone serves a medical center M and five hospitals H1–H5 on a 4 km × 4 km grid with three no-fly zones and
two chargers (C1 at M, C2 elsewhere). During an 8-hour day, orders arrive at random: plasma (M → hospital, must
arrive within 30 min), medicine (M → hospital, 90 min), samples (hospital → M, 60 min) and supplies (hospital →
hospital, 75 min). An order not delivered within twice its limit expires. The drone carries at most 3 boxes / 2 kg,
flies 1 km/min, uses 3.5 % battery per km (+25 % per kg of cargo), keeps a 15 % reserve and charges at 5 %/min.
Three demand levels ("loads"): about 23 / 52 / 70 orders a day (low / medium / high). Real-world motivation:
drone blood delivery in Rwanda (Zipline) cut delivery times and blood wastage (Nisingizwe et al., Lancet Global
Health, 2022). The hard part is dispatching under unknown future orders, not flight control.

## 2. What each milestone did

| Milestone | Main content | Changes and reasons |
|---|---|---|
| M1 (Gym environment) | Gymnasium simulator, A* routes, action masks, Random/EDF/Slack-aware baselines, demo | Charging became one "fly to a charger and charge to 60/80/100 %" action (short charging steps would make credit assignment hard); orders expire at 2× their limit (backlog cannot grow forever); loads calibrated against Slack-aware because EDF was too weak |
| M2 (Neural network) | Set network with attention (62,928 parameters) and an MLP control (370,459); both trained by behaviour cloning (imitation) of Slack-aware and of a rollout planner; FCFS, nearest-first and rollout baselines; HTML replay | Behaviour cloning was a cheap check that the networks, masks and data pipeline can represent a good dispatcher before PPO. Reward changed: late delivery +0.1 fixed (the M1 sliding late reward made a planner chase hopeless orders), expiry −0.1. Observation gained a 7×7 travel-time table (403 → 452 numbers) |
| M3 (Learning agent) | Own masked PPO from scratch; 500 random training maps; expiry penalty −0.5; insertion, scenario rollout and rolling-horizon baselines; a final test of 61 policies registered before it was run | Random maps for generalization; −0.5 because too many orders still expired; the rolling horizon restored as the strongest baseline |
| Final | No new training. Analyses of the existing final test: comparison with the rolling horizon, per map, operations, charging behaviour; learning curves; a new side-by-side video | — |

## 3. The final method

- **Observation (452 numbers):** 80 global features (time, load, battery, cargo, location, flight times, 7×7
  travel-time table), 12 waiting-order slots × 27 features, 3 cargo slots × 16. Orders that can still be on time
  fill the slots first, by deadline.
- **Actions (26):** PICK slot 1–12, DELIVER to one of 6 facilities, CHARGE at C1/C2 to 60/80/100 %, RETURN to M,
  WAIT (≤ 10 min). Each action except WAIT includes its flight; the agent decides when the action ends.
- **Mask:** removes pickups that exceed capacity or finish after expiry, deliveries with no valid box, and flights
  that would leave too little battery to reach a charger above the reserve.
- **Reward (final):** +1 on time, +0.1 late, −0.5 expired, −0.001 per battery %. One day per episode, γ = 1.
- **Set network:** shared encoder for every order slot, shared scorer for the 12 PICK actions (an order's score
  does not depend on its slot), attention of the drone state over orders and cargo (4 heads, width 64), a small
  head for the other 14 actions, a value head. The MLP is the control.
- **PPO:** 16 parallel days × 128 decisions per update, reward ÷ 10, GAE λ 0.95, clip 0.2, 4 epochs, minibatch
  512, Adam 3e-4 → 1.5e-5, entropy 0.01 → 0.001, gradient norm 0.5. 12 million decisions per run (~3.1 h on 2 CPU
  threads), five seeds (11, 22, 33, 44, 55).
- **Checkpoint selection:** checkpoints every 0.5 M decisions; eligible if ≥ 97 % on time at low load and no order
  type > 3 points below Slack-aware at medium/high load, on 20 validation days × 10 validation maps; the best
  medium/high mean is kept. The shipped agent is seed 11 (checkpoint at 10.5 M decisions).

## 4. Baselines (all written by us)

| Baseline | How it works | Why it is in the comparison |
|---|---|---|
| Random | random allowed action | floor |
| FCFS | oldest order first | arrival order as the only priority |
| EDF | earliest deadline first | deadline as the only priority |
| Nearest-first | action that finishes soonest | quickest next step as the only priority |
| Slack-aware | EDF + unload here first + batch orders going where cargo goes + orders that can no longer be on time last ("slack" = time left after serving an order directly) | our best hand-written rule; same speed as PPO; floor in checkpoint selection; base rule of the planners |
| Insertion | rebuilds a route each decision by inserting visible orders | classic route-construction heuristic |
| Rollout | simulates each allowed action, Slack-aware finishes the day, best total reward | standard simulation-based improvement of a rule |
| Scenario rollout | rollout averaged over 16 sampled futures (next 60 min) | anticipating unknown orders |
| Rolling horizon | tries every 3-action sequence, Slack-aware finishes, executes the first action, re-plans | deepest search we could afford; strongest conventional method (bonus comparison) |

They form a ladder: each step adds one kind of reasoning, and decision time grows from ~0.01 ms to ~1.4 s.

## 5. Final test (registered 29 Oct, run once)

- 61 frozen policies: 9 conventional, 41 PPO runs (final agent and variants), 4 M2 imitation networks, 7 charging
  variants. An earlier test (27 Oct) had informed later changes, so a new test with unused days and maps was run.
- Map 0 (the layout of all milestones): 300 new days per load. 30 unseen maps (2000–2029): 10 days each.
- Measures: daily on-time rate (primary), served rate, expired rate, lateness, distance, battery, backlog.
- Statistics: paired differences on identical days, 95 % bootstrap intervals (10,000 resamples). Within ±1 point
  = practically equivalent; non-inferiority margin −2 points against the rolling horizon.

## 6. Key results

On-time rate (%), medium / high load:

| Method | Map 0 | 30 unseen maps | Decision time (ms) |
|---|---|---|---|
| Slack-aware | 79.6 / 63.3 | 82.6 / 66.1 | 0.008 / 0.009 |
| Insertion | 85.7 / 69.1 | 87.8 / 71.3 | 0.91 / 2.4 |
| Rollout | 89.0 / 74.6 | 90.5 / 76.1 | 18.6 / 30.5 |
| Scenario rollout | 89.2 / 76.5 | 90.4 / 77.8 | 149 / 249 |
| Rolling horizon | 92.2 / 79.4 | 93.2 / 80.6 | 564 / 1,373 |
| **PPO (mean of 5 runs)** | **91.8 / 80.8** | **92.7 / 81.7** | **0.24 / 0.23** |

- PPO beats every rule and both rollouts at medium and high load, on all 30 unseen maps.
- Against the rolling horizon: medium −0.39 (map 0) and −0.55 points (unseen), practically equivalent; high +1.35
  and +1.02. PPO wins 6 of 30 unseen maps at medium load and 25 at high load. PPO decides ~2,400× (medium) and
  ~6,100× (high) faster. At low load all methods except Random are 96–100 % on time.
- Operations on unseen maps (medium / high): PPO better for plasma (89.2 vs 87.1 %, 77.2 vs 72.4 %) and less late
  when late (26 vs 43 min, 32 vs 52 min); the horizon flies less (124 vs 134 km, 133 vs 136 km), serves 0.5 points
  more orders.
- Design-choice evidence (registered comparisons): random maps +1.4 / +2.0 on unseen maps, −0.5 / −0.8 on map 0;
  expiry −0.5 lowers expiries by 1.7–2.5 points for −0.4 to +0.1 on-time; set network beats MLP by 1.3–3.0
  points (0.5–2.3 against MLPs trained twice as long) with 1/6 of the parameters; starting PPO from imitation adds
  only +0.4 / +0.8 on map 0 and nothing on unseen maps; PPO beats the M2 imitation networks by 7.6–17.4 points.
- Charging: PPO charges about 20 times a day (rules ~10–14), early (at 54–61 % battery) and starts the peaks with
  83–89 %. Forcing Slack-aware to charge fully or top up before peaks changed its on-time rate by ≤ 0.7 points;
  removing PPO's full-charge actions cost 1.1–1.5 points. A full battery alone does not explain PPO's lead.

## 7. Limitations and lessons

- Simulated world with illustrative parameters; unseen maps come from the same generator; one drone; 12 visible
  orders; planners score days with the M2 reward, PPO trained with the final one; our rolling horizon is a
  depth-3 search, not a published state-of-the-art algorithm.
- Things that changed our conclusions during the project: the strongest baseline (rolling horizon) turned "PPO
  is far better" into "similar quality, much faster"; the 27 Oct test had to be replaced because it informed
  changes; testing on map 0 only would have favoured fixed-map training; imitation as a starting point gave
  almost nothing.

## 8. Where to find things

- `README.md`: install, run the shipped agent, a 5-minute training demo, quick evaluation, full reproduction.
- `scripts/demo.py` (one day, replay page and picture), `scripts/train_ppo.py`, `scripts/evaluate.py`,
  `scripts/locked_test.py` / `locked_report.py` (final test), `scripts/compare_video.py` (side-by-side video).
- `evidence/final_test/` (all final-test results), `evidence/training/` (learning curves), `evidence/latency/`.
- `assets/replay_ppo.html`, `assets/replay_slack.html`: offline replays of the same high-load day.

## 9. Trained models (in `checkpoints/`, ready to run)

| File | What it is |
|---|---|
| `ppo_set.pt` (+ `ppo_set.json`) | the shipped final agent: set network, PPO, 500 random maps, expiry −0.5, seed 11, checkpoint at 10.5 M decisions; the json records its hash, training command and selection |
| `entrants/` (41 files + `manifest.json`) | every PPO policy of the final test: the five final runs (`f12_mm_e05_s11` … `s55`) and all variants (fixed map 0, expiry −0.1, M1 / on-time-only reward, imitation start, MLP 12 M / 24 M) |
| `bc_set.pt`, `bc_mlp.pt` | M2 networks trained by imitating Slack-aware |
| `bc_rollout_set.pt`, `bc_rollout_mlp.pt` | M2 networks trained by imitating the rollout planner |

After the installation in `README.md`, for example:

```bash
python scripts/demo.py --policy ppo --load high --seed 100007                 # shipped agent, one day + replay page
python scripts/demo.py --policy ppo --load high --seed 100007 --map 2000      # the same on an unseen map
python scripts/demo.py --policy ppo --checkpoint checkpoints/entrants/f12_mm_e05_s22.pt --load medium --seed 100007
python scripts/evaluate.py --methods slack,insertion,rollout --checkpoints checkpoints/ppo_set.pt --loads medium,high --days 5 --out outputs/quick_eval.csv
python -m scripts.compare_video --out outputs/compare.mp4                      # side-by-side video, PPO vs Slack-aware
```
