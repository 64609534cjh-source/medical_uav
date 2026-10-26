# Gen2: preregistered new Test-v2 comparison

Test-v2 contains 100 previously unused demand days (300000–300099), at Low/Medium/High, fixed map 0. Day IDs were sealed before architecture development. All 15 selected-best models and statistical source hashes were sealed before Test-v2 was opened. The original Test is retained as historical Generation 1 evidence.

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| m5 | 99.31 ± 0.15 | 84.64 ± 0.29 | 71.11 ± 0.99 | 85.02 ± 0.33 |
| destination | 99.25 ± 0.37 | 85.58 ± 0.52 | 72.89 ± 0.67 | 85.90 ± 0.34 |
| relational | 99.03 ± 0.45 | 84.13 ± 0.80 | 71.75 ± 0.66 | 84.97 ± 0.36 |
| slack | 99.46 | 77.70 | 62.74 | 79.97 |

Rates in percent; ± is sample SD across five independent learners, not a confidence interval.

| Paired contrast | Equal delta (pp) | Joint 95% interval (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| destination - m5 | +0.88 | [+0.41, +1.37] | +0.51, +1.22, +0.23, +1.35, +1.11 |
| relational - m5 | -0.05 | [-0.53, +0.44] | +0.51, +0.25, -0.59, -0.12, -0.30 |
| relational - destination | -0.93 | [-1.48, -0.35] | +0.01, -0.98, -0.83, -1.46, -1.41 |
| m5 - slack | +5.05 | [+4.42, +5.68] | +5.03, +4.73, +5.58, +5.11, +4.82 |
| destination - slack | +5.94 | [+5.34, +6.51] | +5.53, +5.95, +5.81, +6.46, +5.93 |
| relational - slack | +5.00 | [+4.42, +5.58] | +5.54, +4.98, +4.98, +5.00, +4.52 |

### Daily order-type differences from M5

| Type | Method | Low (pp) | Medium (pp) | High (pp) | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | destination | +1.08 | +3.53 | +1.90 | [99, 100, 100] |
| plasma | relational | +1.02 | +0.66 | -1.27 | [99, 100, 100] |
| medicine | destination | -0.10 | -0.49 | -0.22 | [100, 100, 100] |
| medicine | relational | -0.30 | -2.63 | -0.94 | [100, 100, 100] |
| sample | destination | -0.43 | -0.71 | +0.72 | [100, 100, 100] |
| sample | relational | -0.99 | -2.29 | -0.46 | [100, 100, 100] |
| supplies | destination | -0.48 | +3.17 | +5.78 | [99, 100, 100] |
| supplies | relational | -0.44 | +3.57 | +5.64 | [99, 100, 100] |

Daily type rates omit only demand days with no releases of that type. Matched availability is verified across methods. These subtype comparisons are descriptive; no simultaneous-inference claim.

## Interpretation boundaries

The primary contrasts are destination − M5 and relational − M5. The architecture comparisons are pointwise 95% intervals, without familywise correction; treat fine-grained multiple comparisons as exploratory. Paired learner-ID/day-ID product bootstrap uses 10,000 repetitions, common demand-day resampling across loads and separate learner/day-only intervals in the JSON. Five learners do not provide a precise estimate of all training variability.

Only selected-best checkpoints are evaluated here. Training stability and last-checkpoint comparisons use Dev-main/Dev-extra, not Test-v2. No inference about a new map or theoretical optimum is supported. Absence of improvement is a valid result; neither architecture was combined with the other or tuned after this comparison.

5400 full days, 595586 decisions, no illegal action; suite wall time 101.68 seconds. Complete raw per-day metrics and type outcomes remain in `../outputs/gen2_test_v2/`.

Test-v2 is now consumed. Further method selection requires a new holdout or a clearly labeled development comparison.
