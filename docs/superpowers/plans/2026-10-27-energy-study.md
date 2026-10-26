# Energy Intervention Study Plan

**Goal:** Test whether charging ahead of the demand peaks causes the learned policies' peak-time gain, with the four conditions pre-registered in `docs/superpowers/specs/2026-10-27-final-phase-preregistration.md` item 1.

**Architecture:** The interventions are policies in `medical_uav/baselines/energy.py` (Slack-100, Slack-100-prepeak, a full-charge mask for any policy). `scripts/analyze_behaviour.py` runs them with the existing per-day behaviour metrics; `scripts/energy_contrasts.py` computes the registered endpoints from its CSV. Branch `m3-ppo`; no training.

**Fixed before any result:** B = 80 %, W = 30 min, one override per peak, cargo empty (expired boxes count as cargo), the full charge must finish by the peak start (empty-cargo flight energy), earliest finish wins, ties to the lower node index. Selection days first, then screen days; no parameter change after either.

### Task E1: Intervention policies

**Files:** Create `medical_uav/baselines/energy.py`; Test `tests/test_energy.py`

- `slack100_policy(env)`: Slack-aware with charge-target preference (100, 80); everything else unchanged.
- `PrePeakTopUp()`: callable `policy(env)`; before Slack-100's choice, at most one override per peak `p ∈ {60, 300}` when `p − 30 ≤ t < p`, battery < 80, no boxes on board, and a valid `CHARGE(c, 100)` finishes by `p`; keeps `used` (peaks overridden this day), reset when a new day starts.
- `without_full_charge(policy)`: wraps `policy(env, obs, mask)` so `CHARGE(*, 100)` (actions 20 and 23) is removed from the mask it sees; the environment's mask is unchanged.
- Tests: mask removes exactly 20 and 23 and keeps WAIT; Slack-100 equals Slack-aware whenever it does not charge and prefers 100 when it does; override boundaries (t = p − 30 yes, t = p no, battery 80 no, cargo or expired cargo no, finish after p no); one override per peak; reset on a new day.

### Task E2: Metrics and analysis inputs

**Files:** Modify `medical_uav/metrics.py`, `scripts/analyze_behaviour.py`; Test `tests/test_metrics.py`

- `behaviour_metrics` adds `released_peak`, `on_time_peak_count`, `charge_minutes`, `peak_charge_minutes`.
- `analyze_behaviour.py`: methods `slack100`, `prepeak` (rows record `overrides`); `--mask-full-charge` adds a `<label>/no100` twin per checkpoint; `--seeds select|screen`.

### Task E3: Registered contrasts

**Files:** Create `scripts/energy_contrasts.py`; Test `tests/test_energy.py`

- Primary: Δ_timing = mean over medium/high of daily peak-cohort on-time, prepeak − slack100; whole-day bootstrap jointly over the two loads (10 000, seed 0); each load separately.
- Manipulation checks: battery at peak start (≥ +5 points), absolute peak charging minutes (lower), overrides per day.
- Secondary: slack100 − slack, prepeak − slack (all-day lower bound > −1 point for "useful overall"), PPO − PPO-no100 per seed and seed-averaged.
- Verdict printed by the registered table.

### Task E4: Runs and record

- Selection days, then screen days: `analyze_behaviour.py --methods slack,slack100,prepeak --checkpoints <5 selected> --mask-full-charge --loads medium,high --seeds select`, then `--seeds screen`; `energy_contrasts.py` on each; record in roadmap §9 with the registered verdict.
