# M5 pre-tuning diagnostics

Dev-extra 100040–100139 was absent from all 152 recorded local output files before this stage. It is explicitly a development set. Test remains sealed. Original Dev-main selected checkpoints are used unchanged across all loads; no per-load reselection.

| Split / model | Low % | Medium % | High % | Equal % |
|---|---:|---:|---:|---:|
| dev-extra / m4_best | 99.36 | 85.12 | 73.39 | 85.96 |
| dev-extra / m4_last | 99.50 | 83.64 | 70.39 | 84.51 |
| dev-extra / mlp_best | 98.51 | 74.70 | 59.15 | 77.45 |
| dev-extra / mlp_last | 98.62 | 74.12 | 58.23 | 76.99 |
| dev-extra / slack_rule | 99.21 | 79.86 | 65.33 | 81.47 |
| dev-main / m4_best | 99.69 | 85.75 | 74.30 | 86.58 |
| dev-main / m4_last | 99.68 | 83.61 | 71.11 | 84.80 |
| dev-main / slack_rule | 99.26 | 80.99 | 65.89 | 82.05 |

## Matched best-to-last changes on Dev-extra

| Seed | M4 best % | M4 last % | Last − best (pp) | MLP best % | M4 − MLP best (pp) |
|---:|---:|---:|---:|---:|---:|
| 1 | 86.12 | 85.93 | -0.18 | 77.78 | +8.33 |
| 2 | 85.67 | 84.41 | -1.26 | 78.28 | +7.39 |
| 3 | 86.15 | 85.17 | -0.99 | 77.49 | +8.66 |
| 4 | 86.65 | 82.12 | -4.53 | 77.51 | +9.14 |
| 5 | 85.20 | 84.92 | -0.28 | 76.21 | +9.00 |

## Service, failure and transport

Counts pool equal-size day sets across the same five seeds; they retain numerators/denominators rather than averaging conditional ratios. Resource histograms use event weighting and are preserved in JSON.

| Split / load / M4 selection | Reward/day | On-time count | Late count | Unpicked expiry | Picked expiry / pickups | Boxes/effective drop | Batch % | Km/day | Charge min/day | Wait min/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev-main / low / best | 20.972 | 4299 | 14 | 2 | 0 / 4313 | 1.066 | 6.03 | 155.21 | 110.53 | 131.90 |
| dev-main / low / last | 21.110 | 4299 | 15 | 1 | 0 / 4314 | 1.055 | 5.04 | 119.51 | 86.01 | 195.26 |
| dev-main / medium / best | 45.132 | 8830 | 1026 | 456 | 53 / 9909 | 1.321 | 24.31 | 148.75 | 109.00 | 36.21 |
| dev-main / medium / last | 45.497 | 8593 | 1396 | 322 | 54 / 10043 | 1.276 | 21.75 | 149.06 | 109.12 | 40.14 |
| dev-main / high / best | 50.374 | 9969 | 2052 | 1451 | 108 / 12129 | 1.455 | 32.62 | 141.72 | 105.40 | 21.96 |
| dev-main / high / last | 50.805 | 9527 | 2671 | 1272 | 110 / 12308 | 1.427 | 31.23 | 140.61 | 104.91 | 26.09 |
| dev-extra / low / best | 21.627 | 11050 | 78 | 0 | 2 / 11130 | 1.084 | 7.50 | 154.28 | 110.02 | 130.78 |
| dev-extra / low / last | 21.750 | 11064 | 58 | 8 | 0 / 11122 | 1.068 | 6.13 | 121.77 | 88.05 | 188.13 |
| dev-extra / medium / best | 45.073 | 21942 | 2761 | 1147 | 120 / 24823 | 1.319 | 24.11 | 149.29 | 110.08 | 34.88 |
| dev-extra / medium / last | 45.681 | 21531 | 3523 | 798 | 118 / 25172 | 1.288 | 22.17 | 148.40 | 109.21 | 38.15 |
| dev-extra / high / best | 49.889 | 24670 | 5201 | 3806 | 253 / 30124 | 1.447 | 32.55 | 141.93 | 105.94 | 20.44 |
| dev-extra / high / last | 50.407 | 23627 | 6742 | 3292 | 269 / 30638 | 1.412 | 30.63 | 140.67 | 104.95 | 24.71 |

Per-seed/load/type released counts, daily on-time/expiry rates, both expiry failure modes, effective-drop denominators and battery/payload histograms are in the JSON. All new Dev-main scores exactly match the saved M4 records.

M4 LR is constant 3e-4. Target KL 0.02 actually stops remaining joint Actor/Critic updates. Across 25 late checkpoint intervals, the descriptive correlation of interval mean sampled-action final KL and score change is 0.198. This does not establish causality, and does not show the hypothesized negative association. The largest-decline seed is not the one with the largest late KL. M4 did not save full old action distributions, so exact historical KL cannot be recovered.

The primary S1 experiment will alter only the predeclared LR schedule, while adding exact categorical KL and normalized entropy diagnostics without changing KL stopping. There is currently no evidence to prioritize a stricter-KL S2 run. No claim is made that KL can never matter.
