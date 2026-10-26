# Gen2: independent destination and relational architecture experiments

Generation 1 remains frozen at 3c05ad3. Training sources/configurations were sealed before ten scratch runs. Neither new architecture changes observations, actions, reward, PPO, candidate K=12, or the M5 learning-rate schedule. All ten runs complete 1,024,000 transitions, with 11 fixed Dev checkpoints.

## Dev-main selected best

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| m5 | 99.75 ± 0.22 | 86.14 ± 0.62 | 74.26 ± 0.92 | 86.72 ± 0.33 |
| destination | 99.50 ± 0.54 | 86.82 ± 0.39 | 75.73 ± 0.54 | 87.35 ± 0.44 |
| relational | 99.36 ± 0.37 | 85.42 ± 0.90 | 74.76 ± 1.34 | 86.51 ± 0.47 |

Rates in percent; ± is sample SD across five independent learners, not a confidence interval.

| Paired contrast | Equal delta (pp) | Joint 95% interval (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| destination - m5 | +0.63 | [-0.05, +1.33] | +0.30, +1.12, -0.39, +0.85, +1.28 |
| relational - m5 | -0.20 | [-0.96, +0.59] | -0.28, +0.81, -1.09, -0.64, +0.18 |
| relational - destination | -0.84 | [-1.49, -0.22] | -0.57, -0.31, -0.70, -1.49, -1.10 |

## Dev-main last

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| m5 | 99.83 ± 0.19 | 85.51 ± 0.73 | 73.49 ± 1.48 | 86.28 ± 0.66 |
| destination | 99.76 ± 0.19 | 86.60 ± 0.33 | 74.92 ± 0.67 | 87.09 ± 0.34 |
| relational | 99.54 ± 0.26 | 84.75 ± 1.08 | 73.75 ± 1.44 | 86.02 ± 0.74 |

Rates in percent; ± is sample SD across five independent learners, not a confidence interval.

| Paired contrast | Equal delta (pp) | Joint 95% interval (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| destination - m5 | +0.82 | [+0.12, +1.57] | +0.10, +1.65, +0.76, +0.30, +1.28 |
| relational - m5 | -0.26 | [-1.36, +0.95] | -0.80, +1.85, +0.07, -1.50, -0.91 |
| relational - destination | -1.08 | [-2.02, -0.17] | -0.90, +0.20, -0.70, -1.81, -2.19 |

## Dev-extra selected best

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| m5 | 99.57 ± 0.09 | 85.55 ± 0.65 | 73.13 ± 0.70 | 86.08 ± 0.36 |
| destination | 99.46 ± 0.28 | 86.59 ± 0.78 | 74.44 ± 0.85 | 86.83 ± 0.56 |
| relational | 99.17 ± 0.44 | 85.59 ± 0.39 | 73.23 ± 0.71 | 86.00 ± 0.16 |

Rates in percent; ± is sample SD across five independent learners, not a confidence interval.

| Paired contrast | Equal delta (pp) | Joint 95% interval (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| destination - m5 | +0.75 | [+0.05, +1.42] | +0.07, +1.64, -0.26, +1.15, +1.15 |
| relational - m5 | -0.08 | [-0.51, +0.37] | -0.19, +0.35, -0.53, -0.28, +0.23 |
| relational - destination | -0.83 | [-1.37, -0.29] | -0.26, -1.29, -0.27, -1.43, -0.92 |

## Dev-extra last

| Method | Low | Medium | High | Equal |
|---|---:|---:|---:|---:|
| m5 | 99.63 ± 0.16 | 85.66 ± 0.53 | 72.33 ± 1.16 | 85.87 ± 0.46 |
| destination | 99.59 ± 0.17 | 86.83 ± 0.45 | 74.14 ± 1.26 | 86.85 ± 0.51 |
| relational | 99.33 ± 0.34 | 85.38 ± 0.46 | 72.29 ± 0.94 | 85.67 ± 0.30 |

Rates in percent; ± is sample SD across five independent learners, not a confidence interval.

| Paired contrast | Equal delta (pp) | Joint 95% interval (pp) | Seed deltas (pp) |
|---|---:|---:|---|
| destination - m5 | +0.98 | [+0.53, +1.41] | +0.81, +1.47, +0.37, +1.09, +1.15 |
| relational - m5 | -0.20 | [-0.84, +0.43] | -0.75, +0.74, +0.11, -1.03, -0.09 |
| relational - destination | -1.18 | [-1.85, -0.55] | -1.56, -0.73, -0.26, -2.12, -1.24 |

### Daily order-type differences from M5

| Type | Method | Low (pp) | Medium (pp) | High (pp) | Eligible days Low/Med/High |
|---|---|---:|---:|---:|---|
| plasma | destination | +0.46 | +2.54 | +1.28 | [100, 100, 100] |
| plasma | relational | -0.12 | -0.68 | -2.38 | [100, 100, 100] |
| medicine | destination | -0.14 | +0.32 | +0.89 | [100, 100, 100] |
| medicine | relational | -0.17 | -1.11 | -0.76 | [100, 100, 100] |
| sample | destination | -0.22 | -0.69 | -1.03 | [100, 100, 100] |
| sample | relational | -0.75 | -1.29 | -1.17 | [100, 100, 100] |
| supplies | destination | -0.55 | +3.63 | +5.23 | [98, 100, 100] |
| supplies | relational | -0.76 | +4.42 | +6.08 | [98, 100, 100] |

Daily type rates omit only demand days with no releases of that type. Matched availability is verified across methods. These subtype comparisons are descriptive; no simultaneous-inference claim.

## Top-12 diagnostic (Dev-extra, selected best)

| Method | Load | States >12 pending | Hidden same-destination partner | Direct-on-time hidden partner | Maximum pending |
|---|---|---:|---:|---:|---:|
| m5 | low | 0.00% | 0.00% | 0.00% | 7 |
| m5 | medium | 0.54% | 0.48% | 0.10% | 17 |
| m5 | high | 6.63% | 6.35% | 1.89% | 26 |
| destination | low | 0.00% | 0.00% | 0.00% | 6 |
| destination | medium | 0.54% | 0.50% | 0.13% | 18 |
| destination | high | 6.88% | 6.65% | 2.08% | 27 |
| relational | low | 0.00% | 0.00% | 0.00% | 8 |
| relational | medium | 0.50% | 0.47% | 0.13% | 17 |
| relational | high | 6.89% | 6.66% | 2.13% | 26 |

Hidden partner means destination matches visible pending or physical cargo. Direct-on-time is a one-order time-window check, without a full capacity/energy route feasibility test. Correlations and state frequencies indicate exposure, not causal performance loss. They do not establish that K=24 will improve service. No K=24 or combined network is trained.

## Reproducibility

- Training audit: `gen2_training_audit.json`; complete configurations and source hashes: `../configs/gen2_protocol.json`.
- Best is selected only on 40 Dev-main days × three loads. Dev-extra is development, not independent Test. Last is retained without selection.
- Per-day differences, reward, delivered/expired metrics, learner/day-only intervals, type denominators and all curves are in `gen2_development.json`.
- Five learner IDs and demand-day IDs are paired across methods. Intervals use 10,000 product-bootstrap repetitions; day weights are shared across loads. Intervals are pointwise, without familywise correction. Only five learners and one fixed map limit generalization.
- All three methods, regardless of development ranking, remain predeclared Test-v2 candidates. Test-v2 is opened only after hashes of all 15 models and final statistical code are frozen.

## Architecture references

The designs test specific inductive biases; they are not exact reproductions of these papers and the papers do not guarantee benefits in this simulator. [Deep Sets](https://arxiv.org/abs/1703.06114), [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).

## Delivery mechanism diagnostics (Dev-extra best)

| Method | Mean daily batch DELIVER fraction | Pooled boxes / effective DELIVER | Pooled expired-after-pick / picked |
|---|---:|---:|---:|
| m5 | 21.47% | 1.327 | 0.56% |
| destination | 24.01% | 1.347 | 0.28% |
| relational | 21.29% | 1.311 | 0.24% |

Descriptive associations under each learned policy, not causal attribution to a network component. Effective DELIVER excludes actions delivering no box.
