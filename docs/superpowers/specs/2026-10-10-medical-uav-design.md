# Dynamic Medical Pickup-and-Delivery with a Single UAV — Design

Group 17 (Jihang Cai, Yu Liu, Yufeng Zhang) · NUS robotics RL module · design date 2026-10-10 · **revised 2026-10-11 after the M1 review (v2)**

This document is the agreed design for the whole project. It was produced in a three-way discussion (student, Claude, Codex) and replaces the earlier AI-drafted proposal details where they differ. Section 14 records what changed from the proposal and why; section 15 records the decisions taken in the discussion, including the v2 revision.

---

## 1. Intent

A single delivery drone serves one medical center and five hospitals over one operating day. Orders (plasma, medicine, lab samples, inter-hospital supplies) appear at random times, with peaks during the day. Each order type has its own weight and time limit. The drone carries up to 3 boxes / 2 kg and has a battery that must be recharged at one of two charging stations.

The agent decides, at every decision point, **which order to pick up next, where to deliver, when and where to recharge, and when to wait**. A* handles the geometry of each flight. The aim is a learned policy that serves more orders on time than conventional dispatching rules under identical maps and order streams, and an honest analysis of where it does and does not help.

**Success** = on the same map and the same order seeds, the RL policy is compared fairly against conventional baselines with statistically sound evidence, plus behavioural analysis that explains *why* it wins or loses. A loss against the strongest planner, honestly analysed, is still an acceptable outcome; a broken or unexplained result is not.

## 2. Goals, non-goals, anti-goals

**Goals**
- A Gymnasium environment that is correct (tested), deterministic under seeds, readable, and visualised.
- Conventional baselines from simple (FCFS/EDF) to strong (Slack-aware heuristic, rollout planner).
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
- Own conda environment `medical-uav-rl` (Python 3.11, conda-forge only; the system Python is 3.14). Libraries: numpy, gymnasium, torch (from M2), matplotlib, imageio(+ffmpeg), pytest. No RL library in the main pipeline (Stable-Baselines3 may be used only as an optional cross-check and must be disclosed).
- The physical and demand parameters define a **synthetic benchmark**, not a validated delivery system; reports say so.

---

## 4. World model

### 4.1 Map
- Grid 16 × 16 cells, 0.25 km per cell (4 km × 4 km). 2–4 static rectangular no-fly zones.
- 7 nodes: center **M** (also charging station **C1**), hospitals **H1–H5**, standalone charging station **C2**. Nodes sit on free cells, minimum spacing 0.75 km; C2 at least 1 km from M.
- 8-connected A* (diagonal cost √2, no corner cutting) between all node pairs at map creation → 7 × 7 matrices of distance (km) and flight time, plus the cell path for rendering.
- **Validity check** (regenerate with next sub-seed if it fails, max 100 tries; the regeneration success rate is logged and re-checked after any map-size change): all pairs connected; from 100 % battery at either charger, flying with 2 kg to any node i, then to any node j, then to the nearest charger leaves ≥ 15 %.
- Main experiments use **one fixed map: `map_seed = 0`** (frozen in M1; its largest A*/straight-line ratio is 2.04, so no-fly zones clearly matter; 200/200 map seeds generate). Random maps are a stretch experiment (§11).

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
- **Load levels (v2, calibrated against the Slack-aware heuristic, gate G1):** low λ = 0.04 (≈ 23 orders/day), medium λ = 0.08 (≈ 47), high λ = 0.12 (≈ 70). λ = 0.10 is kept as a diagnostic point; λ = 0.14 (≈ 81) only as a labelled stress appendix (it hides on-time-feasible orders beyond the 12 slots in 16 % of decisions). Measured on 40 dev seeds with a prototype of Slack-aware: on-time ≈ 0.82 at 0.08, 0.71 at 0.10, 0.63 at 0.12; EDF only 0.26 at 0.10. The v1 choice (0.04/0.06/0.10 from EDF) is superseded: EDF chases hopeless late orders and is a strawman for calibration.
- The full day's orders are generated from `demand_seed` at `reset()` and revealed only at their release times. They never depend on the agent's actions, so every method sees exactly the same order stream for a given seed.
- **Pending pool cap** 30 (orders released but not picked up). An arrival that finds the pool full is rejected.
- **Measured regime (M1):** ≈ 90–110 decisions per day; charging ≈ 23 % of the day; ≈ 5 km flown per delivered order; mean / p95 / max pending pool ≈ 1.7/6/11 at λ 0.06, 5.1/13/23 at 0.10, 7.4/17/25 at 0.12. Medium-load peaks (≈ 9.6 orders/hour) exceed unbatched capacity, so triage and batching matter exactly when demand peaks.

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

Masks are returned as `info["action_mask"]` (bool array of length 26) and via `env.action_masks()`. Baselines and the agent use the same masks. If a masked action is passed to `step` anyway (e.g. by `check_env`'s random sampling), it is executed as WAIT and `info["invalid_action"] = True`; agents and baselines never do this, and the PPO code asserts it. An action outside `Discrete(26)` raises `ValueError`.

## 6. Observation — `Box(shape=(415,))`, float32, roughly normalised

**Pending slots.** 12 pending orders chosen by a two-tier ranking: first the orders that can still be delivered on time (direct completion ≤ deadline), sorted by time-to-deadline; then the late but not yet expired ones, sorted by time-to-expiry (ties: release time, then order id). This keeps a backlog of late orders from hiding fresh orders that can still be on time. Fewer than 12 → zero-padded with `valid = 0`. Orders beyond the 12 stay in the environment and appear as soon as they rank in the top 12. Calibration logs how often an on-time-feasible order is hidden beyond slot 12: 0 % of decisions at λ 0.06, 2.1 % at 0.10, 6.6 % at 0.12 (Slack-aware). "Feasible" is optimistic (direct timing, ignoring other cargo and recharging). Every method sees the same 12 slots.

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
- **Reward vs headline metric.** A late delivery still earns reward (e.g. +0.85 at 1.1 L) but does not count as on time, so return and on-time rate can disagree. The reward stays as designed; checkpoints are selected and headline results stated by on-time rate, and return, served rate, lateness, per-type results and energy are always reported alongside.

---

## 8. Learning agent

### 8.1 Algorithm: PPO with action masking (written by us in PyTorch)
- Masked categorical: invalid logits set to −1e9 before softmax, both when sampling and when computing log-probs/entropy in the update. The mask seen at sampling time is stored in the rollout buffer and reused in the update; the code asserts that no invalid action is ever sent to the environment.
- γ = 1 (finite episodes, ≈ 100 decisions). GAE λ = 0.95, applied **per decision**. Bootstrap from V(s) at a rollout cut-off, zero at true termination, and never propagate GAE across an episode reset.
- 16 environments × 128 decisions = 2048 transitions per update; 4 epochs; minibatch 512; clip 0.2; Adam 3e-4 with linear decay; entropy 0.01 decaying to 0.001; value coefficient 0.5; gradient clip 0.5; advantages normalised per batch. The learner trains on reward / 10 (critic in the same units); evaluation always reports raw rewards. Observations keep the environment's fixed scaling (no running normalisation of one-hots or slots).
- Collection: start with a synchronous list of environments on the CPU (the environment alone runs ≈ 9,600 decisions/s); benchmark against `AsyncVectorEnv` with 8/16 workers and CPU vs GPU inference, end-to-end including updates, before choosing. Never run three 16-worker jobs at once.
- Tests before any long run: unchanged policy gives probability ratio ≈ 1, hand-computed GAE on a toy trajectory, zero probability on masked actions, correct reset/terminal boundaries.
- Budget: a 2 M-decision pilot, at most two targeted alternatives, then **3 independent runs × 5 M decisions** of the chosen configuration. Curriculum: 0.25 M decisions at low load, 0.75 M at medium, then 4 M with λ drawn uniformly from {low, medium, high} per episode. Evaluate on dev seeds every 250 k decisions.
- Checkpoints are selected on dev seeds only, by on-time rate (§7). Evaluation uses the greedy (argmax over valid actions) policy.
- One policy covers all load levels (λ is part of the observation); no per-load models.

### 8.2 Networks and behaviour cloning
- **(a) MLP:** separate actor and critic, each 415 → 256 → 256 (ReLU) → 26 logits / 1 value.
- **(b) Set network:** shared pending-slot encoder MLP(28→64→64); onboard encoder MLP(16→64); global encoder MLP(31→64); one 4-head attention layer in which the global embedding queries the valid slot embeddings plus one learned always-valid "null" token → attention output. **Context = [global embedding, attention output]**, so battery/time/location always reach the heads, even with zero or one valid slot. PICK(k) logits come from a shared scorer on [slot_k embedding, context]; the other 14 logits from an MLP on the context; the value head is an MLP on [context, masked mean-pool of slot embeddings]. Attention masks absent slots only (not battery-infeasible pickups; those are handled by the action mask). Tests: empty and single-slot observations give finite outputs; permuting pending slots permutes the PICK logits accordingly.
- **Behaviour cloning (BC):** ≈ 1,000 Slack-aware episodes on training demand seeds across the three loads, split by whole episodes into train/validation. Report action-family accuracy, closed-loop performance against the teacher, a tiny-dataset overfit check, and MLP vs set network.
- **Two learning variants, both reported and clearly labelled:** (1) **PPO from scratch** — the primary scientific question; (2) **BC → PPO** — the BC policy as initialisation, then the same PPO budget. BC → PPO is never described as learning from scratch, and its extra cost (demonstration collection + supervised epochs) is reported.

---

## 9. Baselines (same environment, masks, information and physics)

**Shared charging rule** for all heuristics (only ever picks valid actions):
1. Charging is *wanted* if battery < θ, or if no service action (PICK/DELIVER) is valid and battery < 80 %. θ is tuned per heuristic on dev seeds from {20, 30, 40} % (Slack-aware: 30 % in the prototype).
2. If charging is wanted, try in order: CHARGE(nearest station, 80), CHARGE(nearest station, 100), CHARGE(other station, 80), CHARGE(other station, 100). Take the first valid one.
3. Otherwise, or if none is valid: the heuristic's service choice if one exists, else WAIT (always valid).

| baseline | rule |
|---|---|
| Random | uniform over valid actions |
| FCFS | serve the oldest order (deliveries and pickups ranked together by release time) |
| EDF | serve the earliest deadline (calibration and demo reference; a weak baseline) |
| Nearest | serve the action with the smallest travel + handling time; ties broken by deadline |
| **Slack-aware (strong heuristic)** | in this order: (1) if boxes for the current node are on board, DELIVER here; (2) load a pending order at the current node whose destination matches onboard cargo (batching); (3) the shared charging rule; (4) among valid PICK/DELIVER, prefer those that can still be on time (direct timing) over hopeless ones, then earliest deadline, deliveries before pickups on ties; (5) WAIT |
| **Rollout planner** (final milestone; strongest conventional reference, not claimed state of the art) | see below |

**Rollout planner** (one-step lookahead over the Slack-aware heuristic):
- At each decision, for every valid action a: copy the simulator **with only the already released orders** (no future arrivals), execute a, then let Slack-aware act until every known order is resolved (delivered or expired; at most 180 simulated minutes). Score = total simulated reward. Execute the best a; ties go to the Slack-aware choice. With deterministic dynamics this can only match or improve on the base heuristic for the known orders.
- Uses exactly the agent's information (released orders, same masks). Budget ≤ 1 s per decision; evaluation is parallelised over CPU cores; measured decision time is reported. If too slow, restrict a to PICK/DELIVER of the top-6 slots plus the Slack-aware action.

## 10. Evaluation protocol

### 10.1 Seeds and splits (one manifest file records them)
- **Map:** `map_seed = 0`.
- **Train:** demand seeds in [0, 99,999] (the environment's default draw).
- **Dev** (calibration, θ tuning, checkpoint choice): seeds 100,000–100,039 per load.
- **Test** (final numbers only, used once at the end): seeds 200,000–200,099 per load, for every method and all three PPO training seeds.
- **Generalization (stretch):** maps 300,000–300,019 × demand seeds 400,000–400,009.

Every method runs on exactly the same (map, demand seed, load) cases.

### 10.2 Metrics (per episode, saved per day, then aggregated)
- **Primary:** on-time rate = on-time deliveries / all released orders (rejected ones included).
- Return; served rate; expiry rate; rejections; mean and 90th-percentile lateness of late deliveries; on-time rate **per order type** (no type may be starved).
- Energy used; charging minutes; charging share = charging minutes / episode duration; number of charges; battery level when charging starts.
- Mean boxes per delivery stop (batching); share of WAIT and RETURN actions; decision time per step; % of decisions with a hidden on-time-feasible order.
- **Stall diagnostics:** *harmful WAIT* (WAIT while a PICK/DELIVER that would finish on time was valid); *idle expiries* (expired while serving it was valid and the agent chose WAIT or RETURN); *energy-blocked decisions* (all PICK/DELIVER masked only by battery while work existed).

### 10.3 Statistics
Paired per demand day. Report the mean ± 95 % bootstrap CI over test days; the paired difference vs each baseline with its bootstrap CI over days; win/tie/loss counts; for PPO, the spread over the 3 training seeds reported separately from day-to-day variation. Results per load level, never only averaged.

### 10.4 Probes (deterministic hand-built scenarios, pass/fail)
Each probe is a fixed state built by a helper in `tests/probes.py`, checked against the greedy (argmax) policy.

| # | scenario | expected | required from |
|---|---|---|---|
| P1 | single urgent plasma order at M, nothing else | PICK it, then DELIVER (no WAIT) | G2 |
| P2 | battery too low for the only pending order, charger reachable | CHARGE, then serve (no WAIT loop) | G2 |
| P3 | three compatible orders at M, same destination | loads ≥ 2 before leaving | G3 |
| P4 | pending slots permuted | the same order is chosen (set network) | G3 |
| P5 | moderate battery shortly before a peak, nothing pending | charges before the peak | report only |
| P6 | long random rollouts | safety invariant never breaks | M1 (env test) |

### 10.5 Behavioural analysis
Timeline plots (actions, battery, onboard boxes, pool size) for typical days; batching vs time of day; charge timing and level; action distribution in peak vs off-peak; a case study where RL and Slack-aware differ.

### 10.6 Success targets (pre-registered)
- **Competitive:** PPO within 3 percentage points of Slack-aware on-time at medium and high load.
- **Stretch:** ≥ 3 points better than Slack-aware on one pre-registered load (medium) with the paired 95 % CI above zero, and no more than 2 points worse elsewhere.
- **Acceptable outcome:** correct learning, a reproducible fair comparison, and an explained gap. The observation contains engineered hints (slack, compatibility, insertion time), so results are described as learning dispatch from engineered features.

---

## 11. Scope tiers

- **Must:** environment + rendering + tests; Random / EDF / **Slack-aware**; PPO from scratch (MLP or set network) with **3 training seeds**; probes P1–P2; paired test evaluation on 3 loads; side-by-side video; all milestone reports.
- **Should:** BC → PPO variant; set network vs MLP; rollout planner; FCFS / Nearest; probes P3–P5; capacity ablation 1/2/3 boxes (retrained).
- **Could:** random-map generalization; stress load 0.14 with a slot-count ablation; P/R and expiry sweeps; RETURN action ablation; hint-feature ablation; interactive HTML replay built from logged events.

Rule: a Could item starts only if all Should items for that milestone are done. Capacity retrains and random maps are dropped before planner validation or reports are sacrificed.

---

## 12. Milestone plan, acceptance criteria and gates

### M1 — Gym environment (10/11 → 10/16) — status 10/11: implemented, 91 tests, reviewed
Remaining before the M1 report (plan: `docs/superpowers/plans/2026-10-11-m1-finalization.md`):
1. Repository hygiene: untrack `.vscode/`; stage explicit paths only; `.gitattributes` export-ignore for `docs/superpowers/` and `docs/reports/`; README must not link excluded files.
2. Configuration validation and cleanup: `EnvConfig.validate()` (positive durations/rates, `wait_max > 0`, `max_boxes ≤ 3`, `n_slots == 12`, finite λ, known load names); remove unused `charge_targets`; derive the time scale 660 = release window + longest expiry; name observation scales; rename `feasible` → `pick_allowed`; drop decorative separators and § references from code comments.
3. Simulator cleanup with regression tests: one dispose helper; shared next-event helpers; assert orders are sorted; tests for unload + expiry + release at one instant with a full pool, a box expiring during another box's unload, and determinism over the full event log.
4. **Slack-aware heuristic** (§9) with tests; θ tuned on dev seeds.
5. Metrics and calibration: per-type on-time and delivered counts; calibration with Slack-aware / EDF / Random reporting hidden-slot % of decisions, mean hidden count, pool p95/max, rejections, with denominators; seed manifest; set loads 0.04/0.08/0.12 (**amended gate G1: medium is "contested" for Slack-aware, on-time ≈ 0.70–0.85; high clearly lower; low ≈ 0.95+**).
6. Demo: stream frames to the writer (no in-memory frame list); validate CLI arguments; detect non-interactive backends for `--live` and fall back; mark a live run closed early as partial; every frame shows policy, load, demand seed and the current action; order-type legend; "+N more" and "sorted by deadline" on the pending list.
7. Installation: README's primary path = conda-forge with `--override-channels` + `requirements.txt`; `environment.yml` keeps explicit pins and a test checks both files list identical pins.
8. Verification: full tests; fresh conda-forge environment; extracted-archive test; Codex review of the changes; student review of the code and a real-screen check of `--live`.
9. After the student's review: fast-forward `main`; at actual submission create the annotated, never-moved tag `m1-submission-v1` and build the zip from it (code + README + install files + tests + scripts + the student's report PDF).

**Acceptance:** `check_env` passes; all tests pass; same seeds → identical event logs; demo runs from a fresh environment; calibration table on the amended G1.

### M2 — Neural network (start right after M1 fixes → 10/30)
- FCFS / Nearest; `evaluate.py` (paired evaluation over the seed manifest, per-day CSV, bootstrap CIs, plots).
- torch added to the environment files (CPU default; CUDA optional).
- MLP and set networks with the §8.2 tests; BC on ≈ 1,000 Slack-aware episodes; learning curves; closed-loop comparison with the teacher.
- **Masked PPO with the MLP running end-to-end by 10/23** on low load, with the §8.1 tests and an end-to-end throughput benchmark.
- Start a small rollout-planner prototype before M2 closes.
- Report (NN C&R): architecture with tensor shapes and parameter counts, losses and optimiser, dataset provenance and episode-level split, train/validation curves, MLP vs set network, action-family errors, closed-loop results, checkpoint + inference command, tests, preliminary PPO, limitations, AI disclosure.

**Gate G2 (10/30), low-load dev seeds (debugging floor):** PPO-MLP on-time ≥ Random + 20 points and ≥ 0.80 (aim 0.90); P1 and P2 pass; masked-PPO tests pass; zero invalid actions. If not → no new features; up to 3 days (to 11/2) of debugging with probes, stall diagnostics and mask/advantage tests; if still failing, BC → PPO becomes the main pipeline, named as such everywhere.

### M3 — Learning agent (10/31 → 11/6)
- 2 M pilot → at most two alternatives → 3 × 5 M runs (curriculum §8.1) for PPO from scratch; BC → PPO with the same PPO budget; probes P3–P5; dev comparison against Slack-aware.
- **Drop date for the set network: 11/4** — if it trails the MLP on medium dev, the MLP is the main agent.
- **Freeze environment and features on 11/6.** Report.

**Gate G3 (11/6), dev seeds:** "competitive" = within 5 points of Slack-aware on medium and high (≈ 0.77 / 0.58). If more than 10 points behind, pick exactly one bounded fallback: (a) more BC → PPO runs (≤ 3 × 5 M); (b) one longer run of the best configuration (≤ 15 M); or (c) keep the result and make the final report an analysis of why. No new environment features after this gate.

### Final (11/7 → 11/20)
| dates | work |
|---|---|
| 11/7–11/11 | rollout planner validated and timed (**drop date 11/11**: otherwise Slack-aware is the strongest baseline); configurations frozen; test evaluation (100 untouched days per load, all methods, all 3 PPO seeds) |
| 11/11–11/14 | capacity ablation 1/2/3 (3 short retrains), then at most one Could item; **all experiments end 11/14** |
| 11/15–11/17 | figures; side-by-side video (learned policy vs Slack-aware on the same day, synchronised by simulated time) paired with the aggregate test results |
| 11/18–11/20 | individual reports (each member writes their own; drafts start earlier); final zip with the trained model; fresh-environment check; buffer |

**Authorship and walkthroughs.** Claude implements the code step by step with Jihang Cai, who reviews it and must be able to explain every module; Codex reviews; the other members do not write code. AI use is declared in every report as the course requires. Because the ≥ 50 %-from-scratch rule is written about copied online code, the student confirms with the instructor that declared AI-assisted code is acceptable under it; git history is the provenance record. Milestone reports are written by the students.

## 13. Risk register

| risk | early signal | mitigation / fallback |
|---|---|---|
| Data loss (single machine, no remote) | — | private remote + independent backup as soon as the student provides an account; push branches and tags |
| PPO stalls in WAIT or never charges (previous failure) | harmful-WAIT / idle-expiry / energy-blocked diagnostics rising; P1/P2 fail | macro actions, γ = 1, finite WAIT, no terminal penalty; probes gate G2; BC → PPO |
| PPO does not beat Slack-aware | dev gap at G3 | report against the pre-registered targets (§10.6); analyse where decisions differ; rollout planner as the reference; an explained loss is acceptable |
| Return improves but on-time rate does not | dev curves diverge | select checkpoints by on-time rate; report both (§7) |
| Broken masks hidden by the WAIT fallback | invalid-action counter > 0 | stored sampling masks; assertion in the PPO loop |
| Partial observability at high load | hidden-slot % rising | disclosed per load; slot ablation only at the stress load |
| Event/energy bugs | test failures, invariant asserts | unit tests per rule, event-log determinism tests, invariant assertion |
| Training too slow | end-to-end benchmark < 2k decisions/s | synchronous vs async benchmark, CPU vs GPU inference, profiling |
| Schedule slip | milestone acceptance not met | scope tiers; Could items dropped first |
| Authorship interpretation | — | student confirms with the instructor; AI declaration in every report |
| Report rules (no copy-paste, "what changed") | — | the spec's decision log and git history record changes; each member writes their own final report |

## 14. Changes from the original proposal, and why

| proposal | now | reason |
|---|---|---|
| CHARGE / WAIT advance a short fixed interval | CHARGE to a target level; WAIT until the next event (≤ 10 min) | short steps made decision chains too long; the previous agent never learned to charge |
| MOVE to any node | only RETURN_TO_CENTER; charging stations are reached through CHARGE | fewer actions; keeps the anticipatory repositioning that matters |
| soft deadline with ever-growing waiting cost | service limit L + expiry H; linear late reward; one-time −0.5 on expiry or rejection | bounded backlog, a clear signal, matches the cold-chain intuition |
| all orders visible (unbounded) | top-12 pending slots + aggregate counts; pool cap 30 | fixed Gymnasium shapes; overflow counted as failure |
| PPO + attention from the start | MLP first, set/attention network in M3 | de-risk: get learning working before adding architecture |
| random maps as the main setting | one fixed map; random maps as a stretch | learning first, generalization second |
| insertion heuristic + rolling horizon | Slack-aware heuristic + rollout planner | one strong heuristic plus one planner, both fully specified |
| load levels from EDF | load levels from Slack-aware (0.04/0.08/0.12) | EDF is a strawman; a fair challenge must be defined by the strong baseline |

## 15. Decision log (Claude ↔ Codex)

- **Scope.** Codex first proposed a single cargo type and a fixed map without the weight limit. Claude argued that multiple types with different time limits are the core of the user's problem. Agreed: multi-type from v1, curriculum in training, random maps as a stretch.
- **Capacity.** Codex preferred 2 boxes (simpler planner and routing). Claude argued for 3: it matches the proposal, it is the only setting where the 2 kg limit binds, it creates richer batching decisions, and a 1/2/3 ablation is cheap. **Agreed: 3**, with the ablation.
- **Discount.** Claude proposed γ = 0.998/min. Codex argued for γ = 1 because episodes are finite and nothing distorts preferences between near and far deliveries. **Agreed: γ = 1**, GAE per decision. This also removes within-action discounting.
- **WAIT masking.** Both agreed not to mask WAIT when work exists: waiting can enable batching, and masking would hide reward bugs.
- **Late reward.** Claude's first version dropped to 0.5R at L. Codex pointed out the jump and the "last-second rescue" incentive at expiry. **Agreed:** linear from +1 to −0.5 between L and H.
- **Schedule.** Codex warned that integrating PPO only in November is risky. **Agreed:** minimal masked PPO end-to-end by 10/30 (v2: by 10/23).
- **v2 — load calibration.** Claude's probe showed a 4-rule heuristic beats EDF by 34–45 points (triage of hopeless orders contributes most, batching second). Claude proposed 0.06/0.10/0.14 with 16 slots; Codex proposed 0.04/0.08/0.12 keeping the 12-slot layout. **Agreed: 0.04/0.08/0.12**, 0.10 diagnostic, 0.14 stress only.
- **v2 — planner.** Codex found that the v1 plan scoring rewarded doing nothing (the empty plan valued every order as if served later). **Agreed:** a planner whose candidate actions are completed by the base heuristic on known orders. Claude chose the simplest member of that family, the one-step rollout over all valid actions until all known orders resolve (no horizon cap, so no unresolved-order estimate is needed). *To be confirmed by Codex (its round-11 reply was cut off by its usage limit).*
- **v2 — reward vs metric.** Codex flagged that return and on-time rate can disagree. **Agreed:** keep the reward, select and headline by on-time rate, report both.
- **v2 — BC.** Claude wanted BC → PPO as a first-class variant; Codex had treated it as a fallback that does not show independent learning. **Decision (Claude, pending Codex confirmation):** both variants are trained and reported, clearly labelled, with PPO from scratch as the primary question and BC's extra cost reported.
- **v2 — environment files.** Codex suggested `-r requirements.txt` inside `environment.yml`; Claude kept explicit pins in both (the course asks for an exact `environment.yml`) plus a test that they match. *Pending Codex confirmation.*
- **v2 — repository and demo.** Agreed: private remote backup (needs the student's account), immutable annotated submission tags, export-ignore for design/report drafts, streamed animation writing, labelled frames, CLI validation, real-screen check by the student.

## 16. Left to the builder

Exact network widths within the stated ranges; plotting style; file-internal structure; rollout-planner speed-ups within §9; curriculum switch details (decided in M3).

## 17. Repository and submission layout

```
medical_uav/
  env/        config.py, physics.py, astar.py, map_gen.py, orders.py, simulator.py,
              actions.py, masks.py, observation.py, uav_env.py, render.py
  baselines/  heuristics.py, rollout_planner.py
  agents/     networks.py, ppo.py, bc.py
  metrics.py
scripts/      demo.py, calibrate.py, train.py, evaluate.py, make_video.py
tests/        one test file per module, probes.py
models/       selected trained checkpoints (small; committed) with config + seed + checksum
docs/         superpowers/ (spec, plans) and reports/ — kept in git, export-ignored from zips
environment.yml, requirements.txt, README.md, .gitattributes
```

Each milestone is developed on its own branch, merged into `main` after the student's review, and submitted from an annotated tag that is never moved. Submission zips are built from the tag with `git archive` (so editor files, outputs and drafts are excluded) and checked by extracting and running the tests. Videos and large outputs stay out of git and are attached to the final zip.
