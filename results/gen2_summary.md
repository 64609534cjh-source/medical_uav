# Gen2 — Destination-conditioned DELIVER and relational attention

The destination-conditioned head improves equal-load Test-v2 on-time rate by **+0.88 pp** over frozen M5 (pointwise paired 95% interval **[+0.41, +1.37] pp**), with all five paired learner differences positive. The tested single-layer relational model has no clear overall gain (**−0.05 pp**, **[−0.53, +0.44] pp**) and higher inference cost. These are results for the specified implementations and budget, not a general ranking of all attention models or evidence of theoretical optimality.

Retain M5 as the first-generation formal reference and the destination head as the supported second-generation upgrade. The tested attention model does not justify replacing M5. No learner seed is selected using Test-v2.

## Scope

Generation 1 (`3c05ad3`) remains frozen. This stage tests two separate architecture changes, preserving the 403-dimensional information, 26 actions, K=12, simulator, reward, PPO and M5 late learning-rate schedule. Each new method has five scratch learners at 1,024,000 transitions. No combined network, expanded candidates or hyperparameter search was added.

## New Test-v2: selected-best checkpoints

| Method | Low (%) | Medium (%) | High (%) | Equal (%) |
|---|---:|---:|---:|---:|
| m5 | 99.31 ± 0.15 | 84.64 ± 0.29 | 71.11 ± 0.99 | 85.02 ± 0.33 |
| destination | 99.25 ± 0.37 | 85.58 ± 0.52 | 72.89 ± 0.67 | 85.90 ± 0.34 |
| relational | 99.03 ± 0.45 | 84.13 ± 0.80 | 71.75 ± 0.66 | 84.97 ± 0.36 |
| slack | 99.46 | 77.70 | 62.74 | 79.97 |

± is the sample SD across five learner seeds. The 100 new demand IDs 300000–300099 were sealed before development; all 15 models and statistical code were frozen before testing.

| Primary contrast | Equal delta (pp) | Paired joint 95% interval (pp) | Positive learner pairs |
|---|---:|---:|---:|
| destination − m5 | +0.88 | [+0.41, +1.37] | 5/5 |
| relational − m5 | -0.05 | [-0.53, +0.44] | 2/5 |

The destination differences are −0.06 pp at Low, +0.94 pp at Medium and +1.78 pp at High; the corresponding pointwise paired intervals are [−0.49,+0.27], [+0.08,+1.72] and [+0.86,+2.64] pp. Low does not show a clear advantage. These new demand days differ from the first-generation Test: the identical frozen M5 weights score 85.02% here versus 86.42% on the old Test. Cross-cohort score subtraction is not a measure of training degradation.

Intervals resample learner-ID and matched demand-day-ID blocks; common day weights span loads. These are pointwise exploratory intervals without multiplicity correction. Five learners and one fixed map limit broader claims. The full report includes separate learner/day components and all daily differences.

## Development: peak selection and retention

| Method | Dev best | Dev last | Dev last-three mean | Dev-extra best | Dev-extra last |
|---|---:|---:|---:|---:|---:|
| m5 | 86.72% | 86.28% | 86.14% | 86.08% | 85.87% |
| destination | 87.35% | 87.09% | 87.06% | 86.83% | 86.85% |
| relational | 86.51% | 86.02% | 85.93% | 86.00% | 85.67% |

Last-three means use the fixed checkpoints at 819,200, 921,600 and 1,024,000 transitions. Dev-extra is development, not a second independent test. Test-v2 includes best only, so no Test claim about last-checkpoint stability is made.

## Compute cost

| Method | Parameters | Batch 1 p50 / p95 (µs) | Batch 8 p50 / p95 (µs per batch) |
|---|---:|---:|---:|
| m5 | 65,232 | 184.8 / 197.8 | 279.1 / 294.3 |
| destination | 79,675 | 259.1 / 280.2 | 369.0 / 384.7 |
| relational | 102,896 | 347.7 / 381.5 | 577.2 / 623.6 |

Canonical masked categorical argmax, including critic, measured with one CPU thread, identical real observation/mask archive, 200 warmups and 2,000 calls per model/batch. Table entries are medians of the five per-model quantiles; they are not pooled p95s. M5’s separately verified factored action-only inference remains available; this fair architecture comparison uses the canonical interface for all methods.

| New method | Run wall time range (s) | Sum of five run wall times (s) |
|---|---:|---:|
| destination | 1004.2–1031.5 | 5068.2 |
| relational | 1800.6–1830.0 | 9071.1 |

Runs use five concurrent single-thread learner processes. Summed per-run wall time is not the suite elapsed time; the manifest retains actual suite wall time.

## Top-12 exposure and type trade-offs

| M5 load | States >12 pending | Hidden same-destination partner | Direct-on-time hidden partner |
|---|---:|---:|---:|
| low | 0.00% | 0.00% | 0.00% |
| medium | 0.54% | 0.48% | 0.10% |
| high | 6.63% | 6.35% | 1.89% |

These Dev-extra diagnostics use simulator details only after/beside policy inference. A hidden destination match is a possible batching opportunity, not a feasible route certificate. Congested days can already be harder, so correlations do not identify a causal K=12 loss. No K=24 or action-interface expansion was performed.

On Test-v2, destination conditioning increases plasma daily rates by +3.53/+1.90 pp and supplies by +3.17/+5.78 pp at Medium/High. Medicine changes by −0.49/−0.22 pp; sample changes by −0.71/+0.72 pp. These subtype differences are descriptive, with no simultaneous-inference claim. The model does not improve every type or load. In Dev-extra, the mean daily batch-DELIVER fraction rises from 21.47% to 24.01%, and pooled expired-after-pick/picked falls from 0.56% to 0.28%; these are policy behavior associations, not causal module attribution.

All four order types and three loads are reported in the development and Test-v2 reports, with day denominators. An equal-load gain cannot be described as universal improvement when a type/load declines.

## Verification and artifacts

- Ten complete runs: 10,240,000 transitions, 10,000 PPO updates, 110 Dev checkpoints, no illegal action or nonfinite required training diagnostic/checkpoint parameter.
- Each source archive was verified against its manifest; configs, selection rules, complete days, model hashes and first-generation file hashes were audited.
- Training source commit: `705806aba80a06a992286464e31a363f698d726d`.
- New Test-v2: 5,400 full episodes, 595,586 decisions, no illegal actions.
- [Development detail](gen2_development.md), [Test-v2 detail](gen2_test_v2.md), [learning curves](gen2_learning_curves.png), [focused tail curves](gen2_tail_curves.png), [Test-v2 paired differences](gen2_test_v2_paired.png).
- [Architecture and reproduction notes](../docs/gen2_architectures.md), [training protocol](../configs/gen2_protocol.json), [Test-v2 protocol](../configs/gen2_test_v2_protocol.json).
- Raw checkpoints, complete daily metrics and update logs are retained under `outputs/gen2_architectures`, `outputs/gen2_destination_devextra`, `outputs/gen2_relational_devextra`, `outputs/gen2_m5_devextra` and `outputs/gen2_test_v2`.

No method has been proved theoretically optimal. New-map generalization and an aligned optimality bound remain untested. Test-v2 is now consumed; no architecture or hyperparameter changes followed its results.
