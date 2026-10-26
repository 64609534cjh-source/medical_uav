# M4-A: frozen MLP diagnostic (Dev only)

Random scores reproduce both historical protocols with identical immutable order fingerprints on all 120 Dev days. Legacy: `default_rng(day)`; current: `SeedSequence([learner_seed,day,load_index,map_seed])` (canonical learner seed 0). Both uniformly sample legal actions. This is a documented RNG change, not evidence of environment drift; no PPO retraining is indicated.

## Daily service and reward

| Load / model | On time % | Reward | On-time count | Late count | Expired | Flight energy |
|---|---:|---:|---:|---:|---:|---:|
| low / slack | 99.26 | 21.143 | 21.400 | 0.175 | 0.000 | 355.21 |
| low / best | 98.68 | 20.771 | 21.240 | 0.270 | 0.065 | 603.27 |
| low / last | 98.52 | 20.925 | 21.210 | 0.345 | 0.020 | 479.13 |
| medium / slack | 80.99 | 40.434 | 41.550 | 5.075 | 5.200 | 604.77 |
| medium / best | 74.31 | 38.447 | 38.000 | 7.900 | 5.925 | 664.48 |
| medium / last | 73.68 | 38.926 | 37.635 | 8.610 | 5.580 | 645.22 |
| high / slack | 65.89 | 40.551 | 44.025 | 10.125 | 13.750 | 628.54 |
| high / best | 60.93 | 37.728 | 40.580 | 11.170 | 16.150 | 677.50 |
| high / last | 59.44 | 37.757 | 39.535 | 12.455 | 15.910 | 668.72 |

Slack is ahead in reward as well as on-time rate in every load. Reward/selection mismatch alone cannot explain the gap. PPO best/last are five-seed means of daily means; Slack is the same 40 days per load.

## Seed 4: preceding 25 updates

| Diagnostic | Best (376–400) | Last (976–1000) |
|---|---:|---:|
| learning_rate | 0.0003 | 0.0003 |
| approx_kl | 0.00131508 | 0.00099128 |
| clip_fraction | 0.0129004 | 0.010498 |
| epochs_completed | 4 | 4 |
| optimizer_steps | 16 | 16 |
| value_loss | 1.91537 | 1.75811 |
| explained_variance | 0.986536 | 0.987899 |
| return_mean | 15.0275 | 16.277 |
| return_std | 10.3501 | 10.6743 |
| advantage_std | 1.53323 | 1.44194 |
| entropy | 0.511671 | 0.295281 |

No large KL or loss blow-up appears in these windows; epochs remain four and LR remains 3e-4. This does not establish the cause of the 2.32 pp best-to-last decline. Return increases and entropy falls. Historical training legal counts were not logged; the replay measurements below are Dev-state diagnostics, not a reconstruction of training entropy context. The existing M3 report retains seed 4 service/expiry/reward changes.

## Medium / High behavior

| Load / model | Boxes per drop | Batch drop % | Picked then expired % | Charge arrival battery | Charge target | Hidden orders/state | Entropy/log legal |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium / slack | 1.270 | 22.34 | 1.95 | 28.69 | 80.14 | 0.046 | n/a |
| medium / best | 1.160 | 12.83 | 1.64 | 36.03 | 68.66 | 0.070 | 0.199 |
| medium / last | 1.184 | 14.60 | 1.72 | 32.54 | 67.27 | 0.076 | 0.157 |
| high / slack | 1.413 | 32.88 | 3.17 | 26.42 | 80.14 | 0.657 | n/a |
| high / best | 1.161 | 13.62 | 2.60 | 31.81 | 67.53 | 0.796 | 0.214 |
| high / last | 1.190 | 15.38 | 3.07 | 28.42 | 65.63 | 0.807 | 0.164 |

MLP best shows fewer multi-box deliveries and lower charging targets than Slack, but also less expiry after pickup. No waiting action occurred while a service action was legal in these replayed runs. Hidden-order exposure is higher for MLP at High. These associations motivate the shared-order representation experiment; they do not identify an action-level causal error because policies visit different states.

All 880 behavior replays exactly reproduce the saved on-time rate and reward. Event-weighted denominators, payload, charging context, pickup slack, action/time and per-type outcomes remain in `outputs/m4a_diagnostics/behavior_days.jsonl`; complete aggregates and all seed windows are in `results/m4a_diagnostics.json`. No Test day was opened.
