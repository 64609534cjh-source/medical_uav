# Final test: pre-registered design (2026-10-29)

Written before any result of the entrants below on these days or maps. It supersedes the locked test of 10/27 (`2026-10-27-final-phase-preregistration.md` item 3) as the primary final evaluation. That test ran once; its results then informed development (the expiry −0.5 reward, multi-map training), and its entrants, trained to 6 M decisions, were found still improving, which motivated the 12 M budget. Its manifest, checkpoints and results stay in git history; the final report states that it existed and that this test uses new frozen entrants on untouched days and maps.

## Days and maps

Never used by any training, tuning or selection before this test. The only check run on them was that the 30 maps generate (feasibility only, no policy run); all 30 did, so no map was replaced. Map 0 is held out in demand, not in geography; only the second block tests unseen maps.

| Block | Map | Demand days | Loads |
|---|---|---|---|
| **Map 0** | 0 (the map of every earlier result) | 800000–800299 | low, medium, high |
| **Unseen maps** | 2000–2029 (30 maps; all generate) | map 2000+i uses days 800300+10i … 800309+10i | low, medium, high |

Environment v2 (observation with the travel table, v2 metrics). Every entrant runs every day of both blocks (1,800 days), except the rolling horizon, which runs medium and high only (1,200 days): at low load every method is close to 100 % on time, and the horizon is the slowest method.

## Entrants

All PPO runs use the unified budget of 12 M decisions, the set network unless stated, training seeds 11/22/33/44/55 (controls 11/22/33), each run's checkpoint chosen by the roadmap §4 rule: single-map runs on the 100 selection days of map 0, multi-map runs on the first 20 selection days of each selection map (100–109). The two selection domains differ, so single- against multi-map contrasts compare the whole training-and-selection procedure. The 24 M MLP runs are an extra-compute control, added because the 12 M MLP curves still rose by 0.9 points over their last 2 M decisions (20 selection-day curve, medium/high mean, three seeds); a 24 M run has its own learning-rate and entropy schedule and is a separate experiment, not a continuation. The shipped agent is the final agent's seed 11, fixed now.

| Group | Entrants |
|---|---|
| **Final agent** | multi-map, expiry −0.5 (`f12_mm_e05`) × 5 |
| **Development path (2 × 2)** | multi-map v2 (`f12_mm_v2`) × 5; single-map expiry −0.5 (`f12_sm_e05`) × 5; single-map v2 (`f12_sm_v2`) × 5 |
| **Controls** | MLP v2, 12 M (`f12_sm_v2_mlp`) and 24 M (`f24_sm_v2_mlp`) × 3 each; Slack-BC → PPO, expiry −0.5 (`f12_sm_e05_bc`) × 3; M1 reward (`f12_sm_m1`) × 3; on-time-only reward (`f12_sm_ontime`) × 3 |
| **Imitation only** | `bc_set`, `bc_mlp` (Slack-aware teacher), `bc_rollout_set`, `bc_rollout_mlp` |
| **Rules** | Random, EDF, FCFS, nearest-first, Slack-aware, insertion (`late="end"`, chosen on the screen days) |
| **Planners** | visible-order rollout, scenario rollout (16 scenarios), rolling horizon (visible information, depth 3) |
| **Energy study** | Slack-100, Slack-100-prepeak; final agent with `CHARGE(*,100)` hidden × 5; contrasts and readings as registered in the 10/27 file, item 1 |

The foresight horizon (real future orders) stays a development reference: it is not a dispatcher.

## Measures

Per day: on-time, served and expired rate (denominator: released orders), mean lateness (over late deliveries only), energy, km flown, empty km, mean backlog, per-type released and on-time counts (pooled rates recomputed from the counts), decision times (raw samples kept for an overall p95). Primary: the mean daily on-time rate per block and load.

## Statistics

Paired differences on the same days; 95 % percentile intervals from 10,000 bootstrap resamples, seed 0. Five-seed (three-seed) entrants are averaged per day first; controls built on seeds 11/22/33 are also compared seed by seed. Map 0: resample days. Unseen maps: average each map's paired differences over its 10 days, then resample the 30 map means with equal weight; per-map differences and the number of maps won are descriptive. The intervals are conditional on the frozen trained policies, not on future training runs; per-seed means, SD and range are reported for every PPO group. Intervals are per comparison.

## Comparisons and readings (fixed now)

Readings for a difference: *better* if the interval is above 0, *worse* if below 0, otherwise no direction; separately, *practically equivalent* if the interval lies within ±1 point. Non-inferiority at a margin is a third, separate reading.

1. **Primary: final agent against Slack-aware and against the rolling horizon**, each block, medium and high. Roadmap §1 targets, kept: against Slack-aware a medium-load gain of at least 3 points with the interval above 0, and non-inferiority at −2 (high) and −1 (low); against the rolling horizon non-inferiority at −2 points. Served and expired rates are reported beside the on-time rate for every comparison.
2. **Secondary: final agent against every other rule and planner**, each block and load.
3. **Development path.** Map effect: `mm − sm` at each reward, unseen maps (primary: gain ≥ 1 point with interval above 0) and map 0 (non-inferiority −1). Reward effect: `e05 − v2` at each map setting, on-time (non-inferiority −1) and expired rate (lower is the aim).
4. **Controls.** Set − MLP at 12 M (equal budget) and set (12 M) − MLP (24 M, extra compute); BC → PPO − scratch (`f12_sm_e05`) and − `bc_set` (its imitation start); `f12_sm_v2` − M1 reward and − on-time-only.
5. **Transfer gap.** Each PPO group and every rule: unseen-map score minus map-0 score. The blocks use different days, so this is descriptive, not a paired estimate.
6. **Decision time** of every entrant: mean and p95 per decision, timed in the run and in a separate serial benchmark.

## Freeze and run

`locked_test.py freeze` copies the selected checkpoints to `checkpoints/entrants/` and writes the manifest with their SHA-256 hashes, this file and the code commit; the 10/27 manifest and results stay in git history. This file, the selection outcomes and the report script are committed before the test opens. Rehearsal on the selection days first (enumeration, resume, report); then one run of the test blocks into a new output directory, one JSON file per entrant-day, resumable. No entrant is changed, added or reselected after any test result; a bug found afterwards is disclosed, with the corrected rerun limited to the affected entrants.
