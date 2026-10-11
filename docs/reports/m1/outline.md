# M1 report outline (2 pages)

Material for writing the "OpenAI Gym Code and Report" submission. Write the text yourselves; the
points below are facts to draw from.

## 1. Code explanation

- Layers: `map_gen` (grid, no-fly zones, A* matrix) -> `orders` (seeded daily order stream) ->
  `simulator` (event-driven day: pick / deliver / charge / return / wait) -> `masks` + `observation`
  -> `uav_env.MedicalUAVEnv` (Gymnasium API) -> `render` (replay of the simulation history).
- One `step` = one macro action, which advances the clock by its real duration (flight, handling,
  charging, waiting). Releases and expiries that happen inside the action are processed at their
  exact times; same-time order: unload -> expiry -> release.
- Safety invariant: after every action, battery >= 15 % + energy to the nearest charger. The mask
  preserves it, so the drone can never be stranded; asserted in code and tested.
- Determinism: map from `map_seed`, orders from `demand_seed`, so every policy sees the same day.
- Figures: `figures/snapshot_slack_medium_3_t*.png` (Slack-aware baseline, medium load, demand seed 3;
  the header shows the action being executed).

## 2. Existing code / libraries

- numpy, gymnasium (API + `check_env`), matplotlib, imageio, pytest. No external simulator,
  environment or RL code.
- AI assistance (Claude, Codex) declared; see `ai_declaration.md`.

## 3. Reflections / changes from the proposal

- CHARGE/WAIT no longer advance a fixed short interval: CHARGE goes to a station and charges to a
  target (60/80/100 %), WAIT lasts until the next order event. An earlier attempt with short steps
  produced very long decision chains in which the agent stalled or never learned to charge.
- Time limits became two-stage: service limit L (on time) and expiry H = 2L (order lost), with a
  linearly decreasing late reward. This bounds the backlog and gives a clear signal.
- MOVE to any node was reduced to RETURN_TO_CENTER; charging stations are reached through CHARGE.
- The policy sees the 12 most relevant pending orders (on-time-feasible first) instead of all of
  them, so the observation has a fixed size.
- One fixed map for the main experiments; random maps are kept for a later generalization test.
- Calibration (`calibration.md`): EDF turned out to be a weak reference — it keeps chasing orders
  that can no longer be on time and falls far behind at moderate load (0.51 on time at λ 0.08, with
  its charging threshold tuned). A Slack-aware heuristic (deliver at the current node first, batch
  same-destination orders, put hopeless work last) scores 1.00 / 0.84 / 0.65 at λ 0.04 / 0.08 / 0.12,
  so the load levels are defined against it.
  Triage of hopeless orders is the largest single effect, batching the second.
- The environment runs ~9,100 decisions/s per process, ~90–110 decisions per day.
- Next: stronger heuristic baselines, neural network, masked PPO.
