# Frozen final Test comparison

All candidate checkpoints and this analysis protocol were frozen before opening Test. Every learned method uses five independently trained learners and one preselected Dev-main best checkpoint per learner, shared across all loads. No seed or model is selected using Test.

## Main result

| Method | Low % | Medium % | High % | Equal-load % | Joint 95% interval, equal % |
|---|---:|---:|---:|---:|---:|
| Flat MLP | 98.04 ± 0.75 | 76.39 ± 0.97 | 58.72 ± 1.49 | 77.72 ± 0.76 | [76.51, 78.89] |
| Structured S0 | 99.10 ± 0.33 | 86.61 ± 0.81 | 72.94 ± 0.85 | 86.22 ± 0.32 | [85.46, 86.98] |
| Structured S1 | 99.41 ± 0.13 | 87.00 ± 0.50 | 72.84 ± 1.05 | 86.42 ± 0.37 | [85.63, 87.20] |
| Slack-aware | 99.57 | 81.66 | 65.12 | 82.12 | [81.22, 83.04] |
| EDF | 98.27 | 43.68 | 26.33 | 56.09 | [54.60, 57.71] |
| Random | 50.72 | 26.35 | 19.91 | 32.33 | [31.18, 33.54] |

± is sample SD across five learner means, not a confidence interval. Interval estimates use the separately predeclared paired learner/day bootstrap. The final selected structural method is **Structured S1**.

## All learner repetitions

| Method / seed | Low % | Medium % | High % | Equal % |
|---|---:|---:|---:|---:|
| Flat MLP / 1 | 98.32 | 77.30 | 59.43 | 78.35 |
| Flat MLP / 2 | 98.87 | 76.56 | 59.11 | 78.18 |
| Flat MLP / 3 | 97.91 | 76.61 | 58.37 | 77.63 |
| Flat MLP / 4 | 96.85 | 76.74 | 60.32 | 77.97 |
| Flat MLP / 5 | 98.26 | 74.74 | 56.37 | 76.45 |
| Structured S0 / 1 | 98.85 | 87.23 | 73.17 | 86.42 |
| Structured S0 / 2 | 99.55 | 86.88 | 71.47 | 85.97 |
| Structured S0 / 3 | 99.34 | 86.50 | 73.68 | 86.51 |
| Structured S0 / 4 | 99.00 | 87.18 | 73.06 | 86.41 |
| Structured S0 / 5 | 98.77 | 85.26 | 73.32 | 85.78 |
| Structured S1 / 1 | 99.26 | 87.28 | 72.07 | 86.20 |
| Structured S1 / 2 | 99.55 | 86.88 | 71.47 | 85.97 |
| Structured S1 / 3 | 99.49 | 87.49 | 73.89 | 86.96 |
| Structured S1 / 4 | 99.29 | 87.15 | 73.04 | 86.50 |
| Structured S1 / 5 | 99.47 | 86.19 | 73.72 | 86.46 |

## Paired primary comparisons

| Comparison / load | Difference (pp) | Learner SD (pp) | Joint 95% interval (pp) |
|---|---:|---:|---:|
| m4_minus_mlp / low | +1.06 | 0.71 | [+0.41, +1.80] |
| m4_minus_mlp / medium | +10.22 | 0.29 | [+9.11, +11.28] |
| m4_minus_mlp / high | +14.22 | 1.91 | [+12.50, +16.09] |
| m4_minus_mlp / equal | +8.50 | 0.62 | [+7.78, +9.23] |
| s1_minus_slack / low | -0.16 | 0.13 | [-0.45, +0.14] |
| s1_minus_slack / medium | +5.34 | 0.50 | [+4.26, +6.41] |
| s1_minus_slack / high | +7.72 | 1.05 | [+6.45, +8.91] |
| s1_minus_slack / equal | +4.30 | 0.37 | [+3.77, +4.83] |
| s1_minus_m4 / low | +0.31 | 0.27 | [+0.04, +0.66] |
| s1_minus_m4 / medium | +0.39 | 0.53 | [-0.21, +1.05] |
| s1_minus_m4 / high | -0.10 | 0.58 | [-0.81, +0.59] |
| s1_minus_m4 / equal | +0.20 | 0.36 | [-0.17, +0.61] |
| s1_minus_mlp / low | +1.37 | 0.68 | [+0.72, +2.08] |
| s1_minus_mlp / medium | +10.61 | 0.57 | [+9.44, +11.76] |
| s1_minus_mlp / high | +14.12 | 2.22 | [+12.23, +16.23] |
| s1_minus_mlp / equal | +8.70 | 0.96 | [+7.78, +9.65] |

The JSON also retains learner-only and day-only intervals. A common demand-day resample is used across loads and compared policies. The learner-ID resample preserves the declared seed pairings; identical trajectories are not assumed. No calculation treats 1,500 model/load/day records as 1,500 independent training runs. Intervals are pointwise 95%, without a familywise multiplicity correction. With only five training repetitions, the bootstrap is an uncertainty estimate with limited empirical support, not a guarantee.

## Service outcomes on final Test

| Load / method | Reward/day | On-time count | Late count | Unpicked expiry | Picked expiry / pickups | Boxes/effective drop | Flight km/day |
|---|---:|---:|---:|---:|---:|---:|---:|
| low / Flat MLP | 22.562 | 11482 | 222 | 9 | 27 / 11731 | 1.065 | 168.27 |
| low / Structured S1 | 23.029 | 11661 | 76 | 1 | 2 / 11739 | 1.087 | 102.82 |
| low / Slack-aware | 23.017 | 2336 | 11 | 1 | 0 / 2347 | 1.062 | 100.33 |
| medium / Flat MLP | 39.568 | 19494 | 3804 | 2219 | 388 / 23686 | 1.165 | 171.47 |
| medium / Structured S1 | 46.253 | 22353 | 2711 | 753 | 88 / 25152 | 1.316 | 142.71 |
| medium / Slack-aware | 41.368 | 4180 | 589 | 326 | 86 / 4855 | 1.304 | 153.77 |
| high / Flat MLP | 37.298 | 19867 | 6098 | 7571 | 744 / 26709 | 1.165 | 173.60 |
| high / Structured S1 | 50.980 | 24759 | 5670 | 3611 | 240 / 30669 | 1.456 | 139.83 |
| high / Slack-aware | 41.007 | 4419 | 1055 | 1204 | 178 / 5652 | 1.438 | 156.35 |

Counts sum five equal-size Test runs for learned methods and one run for the deterministic rule; these totals have different repetition counts and are shown as explicit outcome denominators, not directly comparable event volumes. Day-level means and rates are the comparison metrics.

| Load / type | MLP on time % | Final on time % | Slack on time % |
|---|---:|---:|---:|
| low / plasma | 95.32 | 97.85 | 99.54 |
| low / medicine | 99.70 | 99.93 | 99.56 |
| low / sample | 97.45 | 99.89 | 99.59 |
| low / supplies | 99.11 | 99.44 | 99.77 |
| medium / plasma | 69.53 | 74.32 | 76.34 |
| medium / medicine | 87.43 | 94.11 | 87.47 |
| medium / sample | 69.55 | 89.58 | 80.66 |
| medium / supplies | 75.45 | 84.34 | 78.64 |
| high / plasma | 53.06 | 56.73 | 57.69 |
| high / medicine | 72.23 | 81.42 | 75.11 |
| high / sample | 52.03 | 77.43 | 64.16 |
| high / supplies | 54.50 | 69.48 | 59.24 |

Type-specific rates average nonempty-type demand days; actual released/picked/outcome counts, conditional post-pick expiry, charging/waiting and resource histograms are retained in JSON. None of these Test subgroups is used for subsequent tuning.

## Reproducibility and scope

Test suite source commit: `8959ab3d88dfd798f440bdf1a4aa8a656ecafd4d`. Protocol SHA256: `1f667f4d2439b9527de027a89c5e4a654094864dd259c2bbb6fd94cdc257effa`.
Test demand-day IDs: 200000–200099, Low/Medium/High, fixed map seed 0. Random is uniform over legal actions with the frozen isolated SeedSequence([0,day,load_index,0]) stream. Slack and EDF were rerun on these Test days; their Dev scores were not reused.
There are 5400 complete Test episodes, 584040 decisions, zero illegal actions and no incomplete day. Every source/model hash matches the frozen protocol.
The tested method uses the canonical training-network forward and the same deterministic masked categorical argmax evaluation as M3/M4. The optional verified inference optimization is reported separately and does not redefine the scientific Test policy.
This validates demand-day generalization under the fixed simulator/map. It does not establish generalization to new maps, real UAV operations or altered rewards.
No model/selection/schedule changes follow these Test results. Further tuning would be a new exploration, and this Test would no longer be untouched.

Methodological context: [RL evaluation with few runs](https://arxiv.org/abs/2108.13264) motivates retaining training uncertainty; [model-selection bias](https://www.jmlr.org/papers/v11/cawley10a.html) motivates the final frozen holdout. The crossed learner/day bootstrap here is explicitly specified in the protocol; it is not claimed to be an exact implementation of the cited benchmark methodology.
