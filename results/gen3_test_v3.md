# Gen3 frozen Test-v3 comparison

100 new demand IDs400000–400099, fixed before development. All three methods and five selected-best models each were predeclared, irrespective of development ranking. Source/model hashes were sealed before opening Test-v3. The original Test and Test-v2 remain historical only.

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| destination | 98.87 ± 0.51 | 86.28 ± 0.48 | 72.86 ± 0.48 | 86.00 ± 0.33 |
| urgency | 99.10 ± 0.28 | 86.92 ± 0.54 | 73.00 ± 0.90 | 86.34 ± 0.40 |
| coordinated | 99.28 ± 0.21 | 86.30 ± 0.68 | 72.19 ± 0.83 | 85.92 ± 0.42 |
| slack | 99.32 | 79.09 | 63.06 | 80.49 |

Rates in percent; ± is sample SD across five learners, not a confidence interval.

| Contrast | Equal delta (pp) | Paired joint 95% (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| urgency - destination | +0.34 | [-0.21,+0.90] | +0.24, -0.30, +1.14, -0.08, +0.68 |
| coordinated - destination | -0.08 | [-0.43,+0.24] | -0.40, +0.09, +0.00, -0.21, +0.11 |
| coordinated - urgency | -0.42 | [-0.95,+0.10] | -0.64, +0.38, -1.14, -0.14, -0.57 |
| destination - slack | +5.51 | [+4.96,+6.09] | +5.35, +5.69, +5.01, +5.85, +5.68 |
| urgency - slack | +5.85 | [+5.24,+6.47] | +5.58, +5.39, +6.16, +5.77, +6.36 |
| coordinated - slack | +5.43 | [+4.83,+6.05] | +4.94, +5.77, +5.02, +5.63, +5.79 |

### All type comparisons (exploratory pointwise intervals)

| Type | Contrast | Low delta [95%] (pp) | Medium | High | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | urgency - destination | +0.15 [-2.03,+2.35] | -0.95 [-4.07,+2.07] | +0.83 [-2.71,+4.33] | [99, 100, 100] |
| plasma | coordinated - destination | +0.96 [-1.85,+3.61] | +0.01 [-5.79,+6.36] | -1.91 [-8.00,+5.28] | [99, 100, 100] |
| medicine | urgency - destination | +0.05 [-0.12,+0.31] | +0.00 [-1.17,+1.21] | -0.62 [-3.35,+2.10] | [100, 100, 100] |
| medicine | coordinated - destination | +0.04 [-0.17,+0.34] | -0.99 [-2.83,+0.70] | -1.89 [-5.31,+1.43] | [100, 100, 100] |
| sample | urgency - destination | +0.66 [-0.08,+1.69] | +2.27 [+0.07,+4.85] | +2.09 [-2.46,+6.01] | [100, 100, 100] |
| sample | coordinated - destination | +0.80 [+0.02,+1.97] | +1.71 [-1.15,+4.53] | +1.63 [-3.19,+6.61] | [100, 100, 100] |
| supplies | urgency - destination | -0.11 [-0.80,+0.55] | +0.56 [-3.27,+4.66] | -1.92 [-8.22,+4.50] | [99, 100, 100] |
| supplies | coordinated - destination | -0.16 [-0.79,+0.48] | -0.78 [-3.96,+2.69] | -0.87 [-5.02,+3.72] | [99, 100, 100] |

Zero-release days omitted consistently; bootstrap day weights renormalized over eligible days per load. Pooled release counts, all subtype means and learner/day components are in JSON. Counts include five policy repetitions of shared demand days, not independent extra days.

## Boundaries

Paired crossed learner/day bootstrap:10,000 repetitions, common day weights across loads, pointwise95% intervals, no multiplicity correction. Five learners limit precision. Raw episodes include all outcomes, order counts, resource metrics and diagnostics. Only best checkpoints are tested; stability/last comparisons use Dev. No new-map or optimality claim is supported.

5400 full episodes, 585562 decisions, no illegal action; suite wall time 120.47s. Test-v3 is consumed; further method development needs a new holdout or explicitly labeled development evidence.
