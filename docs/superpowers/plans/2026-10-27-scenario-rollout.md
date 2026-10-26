# Scenario Rollout Plan

**Goal:** The pre-registered conventional comparator (`docs/superpowers/specs/2026-10-27-final-phase-preregistration.md` item 2): Monte Carlo rollout over Slack-aware with sample-average action values on common random scenarios of future arrivals, 4 vs 16 scenarios, 60-minute arrival horizon, decision p95 ≤ 1 s.

**Architecture:** `medical_uav/baselines/scenario.py`. At each decision: build a planning root from what the agent observes (time, place, battery, cargo, the visible slot orders, the load's arrival rate and the public peak schedule; no history, hidden or future orders), sample S arrival streams in (t, min(t + 60, 480)) from the demand model with the planner's own random streams, then for every allowed action and every scenario run the action and let Slack-aware finish all modelled work; take the best mean total reward, ties to Slack-aware's choice, then the lowest index. The planning copy skips the 452-feature observation.

**Deviations from the registered text:** (1) the planner reads the observable state from the simulator at full precision (time, place, battery, cargo, visible orders with their ids, which fix the slot order) instead of decoding the float32 observation; a test runs it on an object holding only these. (2) Decided before any result: random streams are keyed by (master seed 271828, decision index within the day, scenario index), without an episode counter, so results do not depend on which days are evaluated or in what order. Scenarios 1–4 are the same for S = 4 and 16.

### Task S1: Arrival sampling and planning root

**Files:** Create `medical_uav/baselines/scenario.py`; Test `tests/test_scenario.py`

- `sample_arrivals(rng, t, lam, cfg, first_id, horizon=60.0) -> list[Order]`: Poisson counts per peak/off-peak segment of (t, min(t + horizon, release window)), uniform release times, then type, route and weight as in `generate_orders`.
- `planning_root(env)`: fresh environment state with copies of the visible orders and cargo (original ids, so slots keep their order), current time/place/battery, no history; mask and slots equal the real environment's.
- Tests: counts match the Poisson mean (many draws), releases inside the interval and sorted, none after the release window; root slots and mask equal the real ones; no hidden, future or resolved orders; the real environment is untouched.

### Task S2: The planner

- `ScenarioRollout(scenarios=4, horizon=60.0, seed=271828)`: `__call__(env) -> action`, `scores(env) -> {action: mean value}`; decision counter reset on a new day.
- Tests: scenarios 1–4 equal between S = 4 and 16; information audit on 100 distinct overflowing states (mutating future and hidden orders, demand seed and environment RNG leaves scores and choice unchanged); without arrivals (horizon 0) the values equal the visible-order rollout's; the chosen action's score ≥ Slack-aware's; only valid actions over a whole day; candidate order does not change scores.

### Task S3: Runtime, selection, confirmation

- Serial benchmark, ≥ 1 000 decisions per load, three passes, S = 4 and 16 (p95 ≤ 1 s on every load and pass; aim < 0.8 s); optimise only where profiling shows the need, verified against the reference.
- Selection days: §4 gates; 16 only if it beats 4 by ≥ 0.5 points on the medium/high mean. Freeze; screen confirmation; record in roadmap §9.
