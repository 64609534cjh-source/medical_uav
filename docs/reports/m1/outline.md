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
- Figures: `figures/snapshot_edf_medium_3_t*.png` (EDF baseline, medium load, demand seed 3).

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
- Calibration (`calibration.md`): medium load set to λ = 0.06 (EDF on-time 0.75), low 0.04 (0.99),
  high 0.10 (0.27). The environment runs ~9,650 decisions/s per process, ~100 decisions per day.
- Observed baseline weakness to address in M2: EDF charges at the center before unloading a box
  destined for the center.
- Next: stronger heuristic baselines, neural network, masked PPO.
