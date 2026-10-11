# Dynamic Medical Pickup-and-Delivery with a Single UAV — Design

Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang) · NUS robotics RL module · design date 2026-10-10

This document is the agreed design for the whole project. It was produced in a three-way discussion (student, Claude, Codex) and replaces the earlier AI-drafted proposal details where they differ. Section 14 records what changed from the proposal and why.

---

## 1. Intent

A single delivery drone serves one medical center and five hospitals over one operating day. Orders (plasma, medicine, lab samples, inter-hospital supplies) appear at random times, with peaks during the day. Each order type has its own weight and time limit. The drone carries up to 3 boxes / 2 kg and has a battery that must be recharged at one of two charging stations.

The agent decides, at every decision point, **which order to pick up next, where to deliver, when and where to recharge, and when to wait**. A* handles the geometry of each flight. The aim is a learned policy that serves more orders on time than conventional dispatching rules under identical maps and order streams, and an honest analysis of where it does and does not help.

**Success** = on the same map and the same order seeds, the RL policy is compared fairly against conventional baselines with statistically sound evidence, plus behavioural analysis that explains *why* it wins or loses. A loss against the strongest planner, honestly analysed, is still an acceptable outcome; a broken or unexplained result is not.

## 2. Goals, non-goals, anti-goals

**Goals**
- A Gymnasium environment that is correct (tested), deterministic under seeds, readable, and visualised.
- Conventional baselines from simple (FCFS/EDF) to strong (rolling-horizon planner).
- A self-written PPO agent with action masking that learns a sensible policy.
- Paired evaluation, behavioural analysis, a side-by-side video, and per-milestone reports.

**Non-goals**
- Realistic drone dynamics, wind, 3D flight, real airspace or clinical data.
- Multiple drones or multi-agent RL.
- State-of-the-art RL tricks beyond what the course covers.

**Anti-goals (things that would make the project a failure even if it "runs")**
- Repeating the previous attempt: a large, unreadable codebase with endless diagnostics and an agent that stalls in WAIT or never charges.
- Results that cannot be reproduced from a fresh conda environment.
- Comparisons that are unfair (different information, different constraints, or a strawman baseline only).
- Code the students cannot explain line by line.

## 3. Constraints

- Deadlines: Gym env 10/16, Neural Network 10/30, Learning Agent 11/6, Final 11/20 (no extension).
- ≥ 50 % of code written from scratch; all external code disclosed; AI use (Claude, Codex) declared in every report.
- Readable, commented, student-level Python. Target ≈ 2.5–3.5k lines total.
- All code, comments and documents in English.
- Hardware: 1× RTX A5000 (24 GB), 24 CPU cores.
- Own conda environment `medical-uav` (Python 3.11; the system Python is 3.14). Libraries: numpy, gymnasium, torch, matplotlib, imageio(+ffmpeg), pytest. No RL library in the main pipeline (Stable-Baselines3 may be used only as an optional cross-check and must be disclosed).

---

## 4. World model

### 4.1 Map
- Grid 16 × 16 cells, 0.25 km per cell (4 km × 4 km). 2–4 static rectangular no-fly zones.
- 7 nodes: center **M** (also charging station **C1**), hospitals **H1–H5**, standalone charging station **C2**. Nodes sit on free cells, minimum spacing 0.75 km; C2 at least 1 km from M.
- 8-connected A* (diagonal cost √2, no corner cutting) between all node pairs at map creation → 7 × 7 matrices of distance (km) and flight time, plus the cell path for rendering.
- **Validity check** (regenerate with next sub-seed if it fails, max 100 tries; the regeneration success rate is logged and re-checked after any map-size change): all pairs connected; from 100 % battery at either charger, flying with 2 kg to any node i, then to any node j, then to the nearest charger leaves ≥ 15 %.
- Main experiments use **one fixed map** (`map_seed` chosen once in M1, then frozen). Random maps are a stretch experiment (§11).

### 4.2 Drone
| parameter | value |
|---|---|
| speed | 1.0 km/min, independent of load |
| battery | 0–100 % |
| flight energy | 3.5 %/km × (1 + 0.25 × load_kg) (2 kg → +50 %) |
| reserve | 15 % (battery never drops below this) |
| charging | linear 5 %/min up to 100 %, no connection overhead, cargo may stay on board |
| handling | 2 min per box to load, 2 min per box to unload; no energy |
| waiting / charging energy | 0 |
| capacity | 3 boxes **and** 2.0 kg (both enforced; capacity is a config value) |
| start of day | at M, 100 % battery, empty |

### 4.3 Orders
| type | route | weight (kg, uniform) | service limit L | expiry H | share |
|---|---|---|---|---|---|
| plasma | M → H | 0.6–0.9 | 30 min | 60 min | 0.20 |
| medicine | M → H | 0.3–0.6 | 90 min | 180 min | 0.30 |
| lab sample | H → M | 0.2–0.4 | 60 min | 120 min | 0.30 |
| supplies | H → H′ (H′ ≠ H) | 0.4–0.8 | 75 min | 150 min | 0.20 |

- Hospitals sampled uniformly. Weights chosen so that heavy mixes hit the 2 kg limit (3 plasma ≥ 1.8 kg) while light mixes hit the 3-box limit.
- An order's age starts at release. Picking up does not reset it.
- **Arrivals:** non-homogeneous Poisson over the 480-min release window, rate λ·m(t), with m(t) = 2.0 in [60, 180) and [300, 390) (two peaks) and 0.6 elsewhere. Expected orders per day = 582·λ.
- Initial load levels (to be calibrated in M1, §12 gate G1): low λ = 0.04 (≈ 23 orders), medium λ = 0.07 (≈ 41), high λ = 0.10 (≈ 58).
- The full day's orders are generated from `demand_seed` at `reset()` and revealed only at their release times. They never depend on the agent's actions, so every method sees exactly the same order stream for a given seed.
- **Pending pool cap** 30 (orders released but not picked up). An arrival that finds the pool full is rejected.
- **Expected regime (back-of-envelope, to be verified in M1):** without batching a single order costs ≈ 4–6 km of flight (≈ 5 min), 4 min handling and ≈ 4 min of charging, so roughly 35–47 orders per day is the unbatched capacity and charging takes ≈ 25–30 % of busy time. Peaks at medium load (≈ 8 orders/hour) therefore exceed unbatched capacity: batching and triage matter exactly when demand peaks.

### 4.4 Order life cycle and events
States: `pending → onboard → delivered`, or `pending/onboard → expired`, or `rejected` at arrival.

- **Delivered** when that box's unload finishes. On time iff age ≤ L.
- **Expired** when age reaches H while not delivered. A pending order is removed. An onboard box becomes *expired cargo*: it keeps its weight and box slot and is unloaded automatically (2 min, no reward) at the next facility stop (M or any hospital; not C2).
- **Same-timestamp order of events:** unload completions → expiries → releases. So an unload that finishes exactly at age H counts as a (late) delivery with reward −0.5, which equals the expiry penalty (§7).
- Events that happen inside a macro action (releases, expiries, each unload) are processed at their exact times. New orders never interrupt an action.
- Every expiry time is known in advance (release + H) and flights are deterministic. So masks can predict exactly which onboard boxes will have expired on arrival, and how long their disposal takes.

### 4.5 Episode termination
The episode ends at the first decision point where the release window (480 min) has closed **and** every released order is resolved (delivered, expired or rejected). Every order is resolved by t = 660 at the latest (latest release 480 + longest expiry 180). The clock can pass 660 by at most the length of the one action in progress, during which no reward can occur. No terminal penalty is needed: every order's outcome is already counted.

---

## 5. Actions — `Discrete(26)` with a mask

| index | action | meaning |
|---|---|---|
| 0–11 | `PICK(k)` | fly to the origin of pending slot k (if not already there) and load that box |
| 12–17 | `DELIVER(n)` | n ∈ {M, H1..H5}: fly to n and unload every onboard box for n, most urgent first |
| 18–23 | `CHARGE(s, target)` | s ∈ {C1, C2}, target ∈ {60, 80, 100} %: fly to s and charge until target |
| 24 | `RETURN_TO_CENTER` | fly to M and do nothing else (pre-positioning for upcoming center orders) |
| 25 | `WAIT` | stay until the next order release, the next expiry, or 10 min, whichever comes first (always > 0 min) |

Whenever the drone arrives at a facility, expired cargo is unloaded first (see §4.4).

**Mask rules.** "E(a→b, w)" is flight energy from a to b carrying w kg. "Nearest charger" is the one with the lowest energy from that node with the given load.
- `PICK(k)`: slot valid; capacity and weight are checked against the cargo that remains after expired cargo is disposed at the origin (the origin is always a facility): remaining boxes + 1 ≤ 3 and remaining weight + w ≤ 2.0; the predicted load completion (arrival + disposal of expired cargo + 2 min) is strictly before the order's expiry; battery − E(here→origin, load) − E(origin→nearest charger, remaining load + w) ≥ 15.
- `DELIVER(n)`: at least one onboard box for n that is still non-expired on arrival; battery − E(here→n, load) − E(n→nearest charger, load after unloading and disposal) ≥ 15.
- `CHARGE(s, target)`: arrival battery = battery − E(here→s, load) ≥ 15; target > arrival battery + 5.
- `RETURN_TO_CENTER`: not at M; battery − E(here→M, load) ≥ 15.
- `WAIT`: always valid. This guarantees the mask is never empty.

**Safety invariant** (asserted in code and tested): after every action, battery ≥ 15 + E(here→nearest charger, load). It holds at the start and every mask rule preserves it, so the drone can never be stranded. By the map validity check, from a full battery every delivery is feasible, so charging always restores progress.

Masks are returned as `info["action_mask"]` (bool array of length 26) and via `env.action_masks()`. Baselines and the agent use the same masks. If a masked action is passed to `step` anyway (e.g. by `check_env`'s random sampling), it is executed as WAIT and `info["invalid_action"] = True`; agents and baselines never do this.

## 6. Observation — `Box(shape=(415,))`, float32, roughly normalised

**Pending slots.** 12 pending orders chosen by a two-tier ranking: first the orders that can still be delivered on time (direct completion ≤ deadline), sorted by time-to-deadline; then the late but not yet expired ones, sorted by time-to-expiry (ties: release time, then order id). This keeps a backlog of late orders from hiding fresh orders that can still be on time. Fewer than 12 → zero-padded with `valid = 0`. Orders beyond the 12 stay in the environment and appear as soon as they rank in the top 12. Calibration logs how often an on-time-feasible order is hidden beyond slot 12.

| block | features (count) |
|---|---|
| global (31) | t/660; remaining release time/480; demand multiplier now, in 30 min, in 60 min (÷2); base load λ/λ_high; battery/100; onboard weight/2; onboard boxes/3; location one-hot (7); flight time to each of 7 nodes (÷10 min); energy to nearest charger/100; pending count/30; pending beyond 12 /30; pending count per type (4, ÷10); releases in last 30 min/10 |
| each pending slot (28) × 12 | valid; type one-hot (4); origin one-hot (6); destination one-hot (6); weight; time to deadline/L (negative when late); time to expiry/L; age/L; direct completion time (fly to origin + 2 + fly to dest + 2)/30; slack = time to deadline − direct completion (÷30); same destination as any onboard box; number of pending orders with the same destination/5; origin is the current location; **insertion extra time** (÷30, below); pick feasible (mask bit) |
| each onboard slot (16) × 3 | valid; type one-hot (4); destination one-hot (6); weight; time to deadline/L; time to expiry/L; expired flag; direct delivery time/30 |

Total = 31 + 12 × 28 + 3 × 16 = **415**.

**Insertion extra time.** The provisional route visits the destinations of onboard (non-expired) boxes in earliest-deadline order. With the candidate added, the route becomes: here → candidate origin (+2 min) → onboard ∪ candidate destinations in earliest-deadline order. The feature is the duration difference, handling included. It is a hint for batching, not a plan the agent must follow.

## 7. Reward (undiscounted, γ = 1)

| event | reward |
|---|---|
| box delivered at age a ≤ L | +1 |
| box delivered at age L < a ≤ H | 1 − 1.5 · (a − L)/(H − L) (linear from +1 at L down to −0.5 at H) |
| order expires (pending or onboard) | −0.5 |
| arrival rejected (pool full) | −0.5 |
| flight energy | −0.001 per % consumed |

- R = 1, P = 0.5. The extra value of rescuing a late order rather than letting it expire shrinks smoothly from 1.5 to 0. During peaks, two fresh on-time orders (+2) beat one near-expired rescue, which is the triage behaviour we want. Off-peak, a late delivery is worth doing unless it is so close to expiry that the flight energy costs more than the remaining gain. This is intended: the value of rescuing an order fades to zero at H.
- No pickup bonus, no holding cost, no terminal penalty. Each order contributes exactly once.
- The step reward is the sum of all event rewards that occur during the macro action. `info` also returns `elapsed` (minutes) and an event log for metrics and rendering.
- P/R sweep {0.2, 0.5, 1.0} and expiry multiplier {1.5, 2, 3} are optional sensitivity studies (§11).

---

## 8. Learning agent

### 8.1 Algorithm: PPO with action masking (written by us in PyTorch)
- Masked categorical: invalid logits set to −1e9 before softmax, both when sampling and when computing log-probs/entropy in the update.
- γ = 1 (finite episodes, ≈ 100–250 decisions). GAE λ = 0.95, applied **per decision**.
- 16 environments (gymnasium `AsyncVectorEnv` if profiling shows it is needed). Rollout of 128 decisions per env = 2048 per update; 4 epochs; minibatch 512; clip 0.2; Adam 3e-4 with linear decay; entropy 0.01 decaying to 0.001; value coefficient 0.5; gradient clip 0.5; advantages normalised per batch; returns scaled (e.g. ÷10, or running normalisation; builder's call).
- Budget: 2–5 M decisions per run. Throughput target ≥ 2,000 decisions/s aggregate (measured in M1), i.e. ≤ 45 min per run. Three training seeds for headline results.
- Checkpoints are selected on dev seeds only (§10). Evaluation uses the greedy (argmax over valid actions) policy.
- One policy covers all three load levels (λ is part of the observation); no per-load models.
- **Curriculum:** stage 1 low load → stage 2 medium → stage 3 mixed (λ drawn from {low, medium, high} per episode). Stage switch is triggered by a dev-score plateau or a fixed decision budget, decided in M3.

### 8.2 Networks
- **(a) MLP baseline:** separate actor and critic, each 415 → 256 → 256 (ReLU) → 26 logits / 1 value.
- **(b) Set network (main):** shared pending-slot encoder MLP(28→64→64); onboard encoder MLP(16→64); global encoder MLP(31→64); one multi-head attention layer (4 heads) in which the global embedding queries the valid slot embeddings plus one learned always-valid "null" token (so the attention is well-defined when no slot is valid) → attention output. **Context = [global embedding, attention output]**, so battery/time/location always reach the heads directly, even with zero or one valid slot. PICK(k) logits come from a shared scorer applied to [slot_k embedding, context]; the other 14 logits come from an MLP on the context; the value head is an MLP on [context, masked mean-pool of slot embeddings]. Tests: empty and single-slot observations give finite outputs; permuting pending slots permutes the PICK logits accordingly (§10.4).
- **Behaviour cloning (NN milestone sanity check):** supervised training of both networks to imitate the slack-aware heuristic (§9) on recorded decisions. Report accuracy and rollout performance to show the network can represent a sensible policy. Optional warm start for PPO, reported as an ablation. It becomes part of the main pipeline only if gate G2 or G3 triggers that fallback, and is then named explicitly in every report.

---

## 9. Baselines (same environment, masks, information and physics)

**Shared charging rule** for all heuristics (only ever picks valid actions):
1. Charging is *wanted* if battery < θ, or if no service action (PICK/DELIVER) is valid and battery < 80 %. θ ∈ {30, 40, 50} % is tuned per heuristic on dev seeds.
2. If charging is wanted, try in order: CHARGE(nearest station, 80), CHARGE(nearest station, 100), CHARGE(other station, 80), CHARGE(other station, 100). Take the first valid one.
3. Otherwise, or if none is valid: the heuristic's service choice if one exists, else WAIT (always valid).

| baseline | rule |
|---|---|
| Random | uniform over valid actions |
| FCFS | serve the oldest order (deliveries and pickups ranked together by release time) |
| EDF | serve the earliest deadline |
| Nearest | serve the action with the smallest travel + handling time; ties broken by deadline |
| Slack-aware (strong heuristic) | EDF, but skip orders that can no longer be on time while any on-time-feasible order exists; at an origin with spare capacity, also load pending orders whose destination matches an onboard box or another loaded order |
| Rolling-horizon planner (final milestone; "state-of-the-art conventional" reference) | see below |

**Rolling-horizon planner.**
- **Candidates:** onboard boxes plus the top 6 pick-feasible orders by slack, chosen from the same 12 visible slots the agent sees. The planner gets exactly the agent's information.
- **Plans:** an ordered pickup subset up to the free capacity (≤ 157, including the empty subset) × an order for visiting the resulting destinations (≤ 3! = 6), so ≤ 942 plans. Plans are naturally short (≤ 3 pickups + ≤ 3 delivery stops), so no separate time horizon is needed.
- **Simulation:** each plan is simulated deterministically with the environment's own travel, handling, disposal and energy functions. If the plan would violate the safety rule, a charge to 80 % at the nearest station is inserted before the first leg that would break it; plans that are still infeasible are discarded.
- **Score:** every candidate is valued exactly once. If it is delivered in the plan, it gets its delivery reward at the predicted time. If not, it gets a terminal estimate: the reward it would earn if served directly right after the plan ends, from the plan's end location (which is −0.5 if it would expire first). Flight energy cost is subtracted. Ties go to the shorter plan.
- **Execution:** execute the plan's first action (PICK, DELIVER or the inserted CHARGE), then replan at the next decision. If there are no candidates, apply the shared charging rule, else WAIT.
- **Budget:** ≤ 100 ms per decision; runtime is measured and reported.

The exact enumeration and scoring code is the builder's call within these constraints. Its runtime is measured and reported.

## 10. Evaluation protocol

### 10.1 Seeds and splits
- **Map:** fixed `map_seed` (frozen in M1).
- **Train:** demand seeds drawn at random from [0, 99,999].
- **Dev** (calibration, θ tuning, checkpoint choice): seeds 100,000–100,029 × 3 loads.
- **Test** (final numbers only, touched once per method): seeds 200,000–200,099 × 3 loads.
- **Generalization (stretch):** maps 300,000–300,019 × demand seeds 400,000–400,009.

Every method runs on exactly the same (map, demand seed, load) cases.

### 10.2 Metrics (per episode, then aggregated)
- **Primary:** on-time rate = on-time deliveries / all released orders (rejected ones included).
- Served rate; expiry rate; rejections; mean and 90th-percentile lateness of late deliveries; on-time rate per order type.
- Energy used; charging minutes; charging share = charging minutes / episode duration; number of charges; battery level when charging starts.
- Mean boxes per loaded trip (batching); share of WAIT and RETURN actions; decision time per step.
- **Stall diagnostics** (these detect the previous failure better than raw WAIT share):
  - *harmful WAIT*: WAIT chosen while some PICK/DELIVER that would finish on time was valid;
  - *idle expiries*: orders that expired while a PICK/DELIVER for them was valid at a decision where the agent chose WAIT or RETURN;
  - *energy-blocked decisions*: decisions where every PICK/DELIVER was masked only because of battery while pending/onboard work existed (frequent → the agent is not charging early enough).

### 10.3 Statistics
Paired per seed. Report the mean ± 95 % bootstrap CI over test seeds; the paired difference vs each baseline with CI; win/tie/loss counts; and for PPO, mean and spread over 3 training seeds. Report results separately per load level, never only averaged.

### 10.4 Probes (deterministic hand-built scenarios, pass/fail)
Each probe is a fixed state built by a helper in `tests/probes.py`, checked against the greedy (argmax) policy.

| # | scenario | expected | required from |
|---|---|---|---|
| P1 | single urgent plasma order at M, nothing else | PICK it, then DELIVER (no WAIT) | G2 |
| P2 | battery too low for the only pending order, charger reachable | CHARGE, then serve (no WAIT loop) | G2 |
| P3 | three compatible orders at M, same destination | loads ≥ 2 before leaving | G3 |
| P4 | pending slots permuted | the same order is chosen (set network) | G3 |
| P5 | moderate battery shortly before a peak, nothing pending | charges before the peak | report only (no gate) |
| P6 | long random rollouts | safety invariant never breaks | M1 (env test) |

### 10.5 Behavioural analysis
Timeline plots (actions, battery, onboard boxes, pool size) for typical days; batching vs time of day; charge timing and level; action distribution in peak vs off-peak; a case study where RL and the best baseline differ.

---

## 11. Scope tiers

- **Must** (needed for a complete project): environment + rendering + tests; Random/FCFS/EDF/Slack-aware baselines; PPO (MLP or set network) trained on the fixed map with **3 training seeds** for the headline configuration (runs are independent and fit in parallel on 24 cores); probes P1–P2; paired test evaluation on 3 loads; side-by-side video; all milestone reports.
- **Should:** set network vs MLP comparison; rolling-horizon planner; probes P3–P5; capacity ablation 1/2/3 boxes (retrained).
- **Could:** random-map generalization; BC warm-start study; P/R and expiry sweeps; RETURN action ablation; hint-feature ablation.

Rule: a Could item starts only if all Should items for that milestone are done.

---

## 12. Milestone plan, acceptance criteria and gates

### M1 — Gym environment (10/11 → 10/16)
| day | work |
|---|---|
| 10/11 | repo skeleton, `config.py`, map generation + A* + validity check, order generator |
| 10/12 | event engine (actions, events, expiry, termination) |
| 10/13 | masks, reward, observation; unit tests (event semantics, masks, invariant, determinism) |
| 10/14 | matplotlib renderer (map, no-fly zones, A* paths, drone interpolated along path, side panel: clock, battery, 3 cargo slots, pending list coloured by urgency); `demo.py` (random + EDF, saves gif/mp4) |
| 10/15 | `calibrate.py`: EDF and random over a λ grid → orders/day, km/order, charging share, utilisation, on-time; choose loads and freeze the map; throughput benchmark |
| 10/16 | README, `environment.yml` verified in a fresh env, 2-page report |

**Acceptance:** `gymnasium.utils.env_checker.check_env` passes; all tests pass; same seeds → identical trajectories; the demo runs from a fresh conda env; calibration table produced.

**Gate G1 (10/15):** medium load should be "contested", with EDF on-time roughly 55–85 % and charging share 15–35 %. If not, adjust knobs in this order: λ values → charge rate → map size. Record every change in the change log.

### M2 — Neural network (10/17 → 10/30)
- FCFS / EDF / Nearest / Slack-aware baselines with the shared charging rule; `evaluate.py` (paired evaluation, CSV, plots).
- MLP and set networks; behaviour cloning on Slack-aware demonstrations.
- **Minimal masked PPO with the MLP running end-to-end** on low load: learning curve, comparison with Random and EDF, measured throughput, mask tests in the update.
- Report.

**Gate G2 (10/30), measured on low-load dev seeds:** PPO-MLP mean on-time rate ≥ Random + 20 percentage points **and** ≥ 80 % of EDF's; probes P1 and P2 pass; masked-PPO unit tests pass. If not → no new features; up to 3 days (to 11/2) of debugging with the probes, stall diagnostics and mask/advantage tests. If it still fails on 11/2, switch to the fallback pipeline "BC warm start (Slack-aware) + PPO fine-tuning", and name it as such in every report.

### M3 — Learning agent (10/31 → 11/6)
- Set network + PPO, curriculum, 3 seeds on medium load; probes; first paired dev comparison vs EDF and Slack-aware.
- **Drop date for the set network: 11/4.** If it is not at least on par with the MLP on medium-load dev by then, the MLP becomes the main agent and the set network becomes an ablation.
- **Freeze environment and features on 11/6.** Report.

**Gate G3 (11/6), measured on medium-load dev seeds:** a "wide margin" means PPO's mean on-time rate is more than 10 percentage points below Slack-aware's. In that case pick exactly one fallback with a fixed budget: (a) BC warm start + PPO fine-tuning (≤ 3 runs × 5 M decisions); or (b) one longer run of the best configuration (≤ 15 M decisions); or (c) keep the result and make the final report an analysis of why. No new environment features after this gate.

### Final (11/7 → 11/20)
| dates | work |
|---|---|
| 11/7–11/11 | rolling-horizon planner (**drop date 11/11**: if not working, Slack-aware becomes the strongest baseline); full test evaluation of all methods |
| 11/11–11/14 | capacity ablation 1/2/3 (3 short retrains), then at most one Could item; ablations stop on 11/14 regardless |
| 11/12–11/14 | stretch: random-map generalization, only if G3 passed without fallback by 11/8 |
| 11/15–11/17 | figures; side-by-side video (PPO vs best baseline, same seed) |
| 11/18–11/20 | individual reports (each member writes their own); final zip; fresh-env check; buffer |

**Throughput** is measured twice in M1/M2: environment-only steps/s and end-to-end decisions/s including PPO updates. Training budgets above assume the end-to-end figure.

**Authorship and walkthroughs.** Code is written with AI assistance (Claude, with Codex as reviewer), declared in every report as the course requires. Plagiarism concerns online and other groups' code; none is copied. After each module lands, there is a walkthrough so the students can explain every line, plus a short entry in `docs/CHANGELOG.md` on what changed and why. Milestone reports are written by the students; AI helps with figures, tables and proofreading. Decided 2026-10-10: Claude implements all code step by step with Jihang Cai (who reviews and must be able to explain each module); the other members do not write code.

## 13. Risk register

| risk | early signal | mitigation / fallback |
|---|---|---|
| PPO stalls in WAIT or never charges (previous failure) | harmful-WAIT / idle-expiry / energy-blocked diagnostics rising; probes P1/P2 fail | macro actions, γ = 1, finite WAIT, no terminal penalty; probes gate G2; BC warm-start fallback |
| PPO does not beat strong heuristics | dev gap at G3 | honest reporting; analyse niche cases (peaks, charge timing); planner as the reference; the result is still valid |
| Event/energy bugs | test failures, invariant asserts | unit tests per rule, determinism tests, invariant assertion in `step` |
| Training too slow | benchmark < 2k decisions/s | precomputed matrices, `AsyncVectorEnv`, profiling |
| Load regime uninteresting | G1 numbers out of range | knob order in G1 |
| Schedule slip | milestone acceptance not met | scope tiers; Could items dropped first |
| Report rules (no copy-paste, "what changed") | — | keep `docs/CHANGELOG.md` per milestone; each member writes their own final report |
| AI-use disclosure | — | AI declaration paragraph in every report |

## 14. Changes from the original proposal, and why

| proposal | now | reason |
|---|---|---|
| CHARGE / WAIT advance a short fixed interval | CHARGE to a target level; WAIT until the next event (≤ 10 min) | short steps made decision chains too long; the previous agent never learned to charge |
| MOVE to any node | only RETURN_TO_CENTER; charging stations are reached through CHARGE | fewer actions; keeps the anticipatory repositioning that matters |
| soft deadline with ever-growing waiting cost | service limit L + expiry H; linear late reward; one-time −0.5 on expiry or rejection | bounded backlog, a clear signal, matches the cold-chain intuition |
| all orders visible (unbounded) | top-12 pending slots + aggregate counts; pool cap 30 | fixed Gymnasium shapes; overflow counted as failure |
| PPO + attention from the start | MLP first, set/attention network in M3 | de-risk: get learning working before adding architecture |
| random maps as the main setting | one fixed map; random maps as a stretch | learning first, generalization second |
| insertion heuristic + rolling horizon | slack-aware heuristic + bounded rolling-horizon planner | one strong heuristic plus one planner, both fully specified |

## 15. Decision log (Claude ↔ Codex)

- **Scope.** Codex first proposed a single cargo type and a fixed map without the weight limit. Claude argued that multiple types with different time limits are the core of the user's problem. Agreed: multi-type from v1, curriculum in training, random maps as a stretch.
- **Capacity.** Codex preferred 2 boxes (simpler planner and routing). Claude argued for 3: it matches the proposal, it is the only setting where the 2 kg limit binds, it creates richer batching decisions, and a 1/2/3 ablation is cheap. **Agreed: 3**, with the ablation.
- **Discount.** Claude proposed γ = 0.998/min. Codex argued for γ = 1 because episodes are finite and nothing distorts preferences between near and far deliveries. **Agreed: γ = 1**, GAE per decision. This also removes within-action discounting.
- **WAIT masking.** Both agreed not to mask WAIT when work exists: waiting can enable batching, and masking would hide reward bugs.
- **Late reward.** Claude's first version dropped to 0.5R at L. Codex pointed out the jump and the "last-second rescue" incentive at expiry. **Agreed:** linear from +1 to −0.5 between L and H.
- **Schedule.** Codex warned that integrating PPO only in November is risky. **Agreed:** minimal masked PPO end-to-end by 10/30.

## 16. Left to the builder

Exact network widths within the stated ranges; return-scaling method; plotting style; file-internal structure; rolling-planner enumeration details within §9; curriculum switch criterion (decided in M3); the specific `map_seed` (chosen in M1).

## 17. Repository layout

```
medical_uav/
  env/        config.py, map_gen.py, astar.py, orders.py, uav_env.py, render.py
  baselines/  heuristics.py, rolling_planner.py
  agents/     networks.py, ppo.py, bc.py
scripts/      demo.py, calibrate.py, train.py, evaluate.py, make_video.py
tests/        test_map.py, test_events.py, test_masks.py, test_determinism.py, test_ppo_masking.py
docs/         CHANGELOG.md, reports/, superpowers/specs/
environment.yml, README.md
```
