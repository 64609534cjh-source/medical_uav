# Gen3 development: urgency and pickup/delivery coordination

All prior generations remain frozen. Two separate architecture changes use the same information, K12 actions, reward, PPO, M5 LR schedule and 1,024,000-transition budgets. User confirmed the original overall Dev selection rule; no subtype guardrail is introduced.

## dev_best

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| destination | 99.50 ± 0.54 | 86.82 ± 0.39 | 75.73 ± 0.54 | 87.35 ± 0.44 |
| urgency | 99.68 ± 0.16 | 87.44 ± 0.40 | 75.59 ± 0.55 | 87.57 ± 0.29 |
| coordinated | 99.70 ± 0.26 | 87.03 ± 0.56 | 75.37 ± 0.72 | 87.37 ± 0.36 |

Rates in percent; ± is sample SD across five learners, not a confidence interval.

| Contrast | Equal delta (pp) | Paired joint 95% (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| urgency - destination | +0.22 | [-0.44,+0.91] | +0.44, -0.42, +1.25, -0.23, +0.07 |
| coordinated - destination | +0.02 | [-0.52,+0.54] | -0.27, -0.13, +0.45, -0.15, +0.18 |
| coordinated - urgency | -0.20 | [-0.79,+0.41] | -0.71, +0.30, -0.80, +0.08, +0.11 |

### All type comparisons (exploratory pointwise intervals)

| Type | Contrast | Low delta [95%] (pp) | Medium | High | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | urgency - destination | +0.32 [-1.50,+2.25] | -0.53 [-3.64,+2.83] | +2.06 [-1.31,+5.57] | [38, 40, 40] |
| plasma | coordinated - destination | +0.36 [-1.84,+2.60] | +0.03 [-5.26,+5.91] | +0.36 [-5.94,+7.86] | [38, 40, 40] |
| medicine | urgency - destination | -0.06 [-0.33,+0.00] | -0.40 [-1.95,+1.14] | -1.23 [-3.54,+1.00] | [40, 40, 40] |
| medicine | coordinated - destination | +0.00 [+0.00,+0.00] | -0.47 [-2.84,+1.56] | -1.73 [-5.06,+1.37] | [40, 40, 40] |
| sample | urgency - destination | +0.28 [-0.17,+0.99] | +3.03 [+0.38,+6.19] | +0.90 [-3.74,+4.90] | [40, 40, 40] |
| sample | coordinated - destination | +0.34 [+0.00,+1.04] | +1.85 [-2.61,+6.05] | +1.02 [-4.12,+6.38] | [40, 40, 40] |
| supplies | urgency - destination | +0.06 [+0.00,+0.37] | +0.29 [-4.54,+5.31] | -1.15 [-6.56,+4.12] | [40, 40, 40] |
| supplies | coordinated - destination | -0.09 [-0.64,+0.25] | -1.33 [-5.53,+2.85] | -0.34 [-4.08,+3.67] | [40, 40, 40] |

Zero-release days omitted consistently; bootstrap day weights renormalized over eligible days per load. Pooled release counts, all subtype means and learner/day components are in JSON. Counts include five policy repetitions of shared demand days, not independent extra days.

## dev_last

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| destination | 99.76 ± 0.19 | 86.60 ± 0.33 | 74.92 ± 0.67 | 87.09 ± 0.34 |
| urgency | 99.70 ± 0.13 | 87.21 ± 0.72 | 75.42 ± 0.52 | 87.44 ± 0.37 |
| coordinated | 99.96 ± 0.09 | 86.06 ± 0.78 | 73.86 ± 1.02 | 86.63 ± 0.51 |

Rates in percent; ± is sample SD across five learners, not a confidence interval.

| Contrast | Equal delta (pp) | Paired joint 95% (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| urgency - destination | +0.35 | [-0.32,+1.05] | +0.64, -0.02, +1.25, +0.32, -0.46 |
| coordinated - destination | -0.46 | [-1.01,+0.10] | -0.98, -0.54, -0.13, -0.47, -0.19 |
| coordinated - urgency | -0.81 | [-1.55,-0.04] | -1.63, -0.52, -1.39, -0.80, +0.27 |

### All type comparisons (exploratory pointwise intervals)

| Type | Contrast | Low delta [95%] (pp) | Medium | High | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | urgency - destination | -0.43 [-1.84,+0.87] | +0.89 [-2.78,+4.82] | +4.72 [+0.66,+8.73] | [38, 40, 40] |
| plasma | coordinated - destination | +0.66 [-0.19,+1.86] | +0.47 [-3.49,+4.51] | +1.46 [-2.38,+5.35] | [38, 40, 40] |
| medicine | urgency - destination | +0.00 [+0.00,+0.00] | -0.31 [-2.37,+1.83] | -0.15 [-3.76,+4.19] | [40, 40, 40] |
| medicine | coordinated - destination | +0.00 [+0.00,+0.00] | -1.33 [-3.55,+0.95] | -1.48 [-5.04,+2.00] | [40, 40, 40] |
| sample | urgency - destination | +0.16 [+0.00,+0.60] | +2.81 [-0.34,+6.17] | +0.39 [-2.99,+4.15] | [40, 40, 40] |
| sample | coordinated - destination | +0.16 [+0.00,+0.60] | +0.29 [-3.46,+4.76] | -0.91 [-4.09,+2.00] | [40, 40, 40] |
| supplies | urgency - destination | +0.00 [+0.00,+0.00] | -1.61 [-5.87,+2.65] | -1.36 [-5.14,+2.53] | [40, 40, 40] |
| supplies | coordinated - destination | +0.00 [+0.00,+0.00] | -1.54 [-5.33,+2.84] | -3.49 [-7.61,+1.13] | [40, 40, 40] |

Zero-release days omitted consistently; bootstrap day weights renormalized over eligible days per load. Pooled release counts, all subtype means and learner/day components are in JSON. Counts include five policy repetitions of shared demand days, not independent extra days.

## dev_extra_best

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| destination | 99.46 ± 0.28 | 86.59 ± 0.78 | 74.44 ± 0.85 | 86.83 ± 0.56 |
| urgency | 99.55 ± 0.17 | 87.11 ± 0.24 | 74.50 ± 0.80 | 87.05 ± 0.33 |
| coordinated | 99.71 ± 0.17 | 86.83 ± 0.58 | 74.23 ± 1.09 | 86.92 ± 0.49 |

Rates in percent; ± is sample SD across five learners, not a confidence interval.

| Contrast | Equal delta (pp) | Paired joint 95% (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| urgency - destination | +0.22 | [-0.45,+0.88] | +1.02, -0.58, +0.86, -0.56, +0.39 |
| coordinated - destination | +0.09 | [-0.25,+0.42] | -0.23, +0.11, +0.51, -0.05, +0.12 |
| coordinated - urgency | -0.13 | [-0.81,+0.51] | -1.25, +0.69, -0.35, +0.51, -0.27 |

### All type comparisons (exploratory pointwise intervals)

| Type | Contrast | Low delta [95%] (pp) | Medium | High | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | urgency - destination | -0.26 [-1.66,+1.02] | +0.24 [-2.91,+4.37] | +0.55 [-2.68,+3.71] | [100, 100, 100] |
| plasma | coordinated - destination | +0.23 [-0.86,+1.37] | +0.04 [-5.66,+6.80] | -0.84 [-5.74,+4.82] | [100, 100, 100] |
| medicine | urgency - destination | +0.07 [-0.27,+0.45] | +0.02 [-1.19,+1.30] | -0.86 [-3.14,+1.29] | [100, 100, 100] |
| medicine | coordinated - destination | +0.14 [+0.00,+0.50] | -0.34 [-2.16,+1.42] | -1.62 [-4.35,+0.99] | [100, 100, 100] |
| sample | urgency - destination | +0.12 [-0.37,+0.66] | +1.74 [-0.44,+3.73] | +2.45 [-1.95,+5.88] | [100, 100, 100] |
| sample | coordinated - destination | +0.30 [-0.14,+0.94] | +1.61 [-2.04,+5.19] | +2.43 [-2.56,+7.71] | [100, 100, 100] |
| supplies | urgency - destination | +0.56 [-0.05,+1.57] | -0.33 [-4.41,+3.88] | -2.52 [-9.20,+4.16] | [98, 100, 100] |
| supplies | coordinated - destination | +0.41 [-0.07,+1.26] | -0.62 [-4.01,+3.34] | -1.43 [-5.59,+3.23] | [98, 100, 100] |

Zero-release days omitted consistently; bootstrap day weights renormalized over eligible days per load. Pooled release counts, all subtype means and learner/day components are in JSON. Counts include five policy repetitions of shared demand days, not independent extra days.

## dev_extra_last

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| destination | 99.59 ± 0.17 | 86.83 ± 0.45 | 74.14 ± 1.26 | 86.85 ± 0.51 |
| urgency | 99.66 ± 0.19 | 86.97 ± 0.52 | 74.15 ± 0.72 | 86.92 ± 0.38 |
| coordinated | 99.77 ± 0.09 | 86.45 ± 0.74 | 73.05 ± 1.60 | 86.42 ± 0.69 |

Rates in percent; ± is sample SD across five learners, not a confidence interval.

| Contrast | Equal delta (pp) | Paired joint 95% (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| urgency - destination | +0.07 | [-0.43,+0.61] | +0.28, -0.29, +0.86, -0.51, +0.02 |
| coordinated - destination | -0.43 | [-1.19,+0.21] | -1.69, -0.61, +0.16, -0.45, +0.44 |
| coordinated - urgency | -0.50 | [-1.36,+0.22] | -1.97, -0.32, -0.71, +0.06, +0.42 |

### All type comparisons (exploratory pointwise intervals)

| Type | Contrast | Low delta [95%] (pp) | Medium | High | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | urgency - destination | +0.14 [-1.15,+1.30] | +0.87 [-2.19,+3.94] | +2.40 [-0.30,+5.03] | [100, 100, 100] |
| plasma | coordinated - destination | +0.48 [-0.23,+1.35] | +1.77 [-0.29,+4.15] | +1.35 [-1.46,+4.23] | [100, 100, 100] |
| medicine | urgency - destination | -0.04 [-0.38,+0.09] | -0.32 [-2.18,+1.56] | -1.23 [-4.76,+2.73] | [100, 100, 100] |
| medicine | coordinated - destination | +0.02 [+0.00,+0.13] | -1.19 [-3.14,+0.67] | -2.66 [-7.28,+1.56] | [100, 100, 100] |
| sample | urgency - destination | +0.15 [-0.28,+0.70] | +1.18 [-0.92,+3.58] | +1.62 [-1.28,+4.37] | [100, 100, 100] |
| sample | coordinated - destination | +0.32 [-0.07,+0.91] | -0.20 [-2.70,+2.77] | +0.95 [-2.42,+4.59] | [100, 100, 100] |
| supplies | urgency - destination | +0.08 [-0.27,+0.47] | -1.82 [-5.14,+1.86] | -2.93 [-7.17,+1.68] | [98, 100, 100] |
| supplies | coordinated - destination | -0.08 [-0.50,+0.23] | -1.76 [-4.99,+1.85] | -4.40 [-7.90,-0.40] | [98, 100, 100] |

Zero-release days omitted consistently; bootstrap day weights renormalized over eligible days per load. Pooled release counts, all subtype means and learner/day components are in JSON. Counts include five policy repetitions of shared demand days, not independent extra days.

## Interpretation and artifacts

Dev-extra is development. All three predeclared methods, all five Dev-selected best models each, remain candidates regardless of development ranking. Test-v3 is opened once only after models and statistical code are sealed.

Urgency features first reconstruct physical minutes before cross-type minima; expired physical cargo is counted separately and still retained in the original group embeddings. The coordination variant separately adds the existing 107-dimensional destination group to PICK, including the candidate itself. No urgency+coordination combination or training-objective change is made.

Read-only failure patterns, pooled delivery outcomes, rewards and complete daily paired differences are retained in `gen3_development.json` and raw evaluation outputs. Deadline crossings during an action are observational, not avoidable causal mistakes. All subtype intervals are shown; no multiplicity correction. Five learners and one fixed map limit generalization.
