# Gen3 — urgency preservation and PICK–DELIVER coordination

**Retain the frozen Gen2 destination head as the formal mainline.** Neither new architecture demonstrates a clear equal-load Test-v3 gain: urgency is **+0.34 pp [−0.21,+0.90]**, coordination **−0.08 pp [−0.43,+0.24]** relative to the same reference. The urgency variant is a research candidate, not an established replacement. The engineering result is a verified action-only/context-reuse option: reference CPU batch1 p50 falls from270.4 to212.5µs with the guarded factored path (about21.4%), preserving all validated full-day behavior.

Two separate hypotheses extend frozen Gen2 destination conditioning. No training-objective, reward, PPO, interface, candidate count or budget change is included. All prior results and model weights remain available. The user explicitly kept the original overall Dev selection rule, without subtype guardrails.

## New Test-v3: same-day comparison

| Method | Low (%) | Medium (%) | High (%) | Equal (%) |
|---|---:|---:|---:|---:|
| destination | 98.87 ± 0.51 | 86.28 ± 0.48 | 72.86 ± 0.48 | 86.00 ± 0.33 |
| urgency | 99.10 ± 0.28 | 86.92 ± 0.54 | 73.00 ± 0.90 | 86.34 ± 0.40 |
| coordinated | 99.28 ± 0.21 | 86.30 ± 0.68 | 72.19 ± 0.83 | 85.92 ± 0.42 |
| slack | 99.32 | 79.09 | 63.06 | 80.49 |

± is sample SD across five learners, not an optimality gap or confidence interval. Test-v3 uses100 new demand IDs400000–400099, fixed map0. IDs were sealed before development; all three methods and fifteen best-model hashes were sealed before opening the holdout. Absolute scores should not be subtracted from previous Test cohorts.

| Primary contrast | Equal delta (pp) | Paired joint95% (pp) | Positive learner pairs |
|---|---:|---:|---:|
| urgency − destination | +0.34 | [-0.21,+0.90] | 3/5 |
| coordinated − destination | -0.08 | [-0.43,+0.24] | 3/5 |

Paired crossed learner-ID/day-ID product bootstrap,10,000 repetitions. All subtype comparisons, eligible days, release counts and separate learner/day components are in [Test-v3 detail](gen3_test_v3.md). Intervals are exploratory pointwise95%, without familywise correction. Five learners and a single map limit claims.

## Development and retention

| Method | Dev best | Dev last | Dev last3 | Dev-extra best | Dev-extra last |
|---|---:|---:|---:|---:|---:|
| destination | 87.35% | 87.09% | 87.06% | 86.83% | 86.85% |
| urgency | 87.57% | 87.44% | 87.25% | 87.05% | 86.92% |
| coordinated | 87.37% | 86.63% | 86.74% | 86.92% | 86.42% |

On Test-v3, urgency changes Low/Medium/High by +0.23/+0.64/+0.14pp, and coordination by +0.41/+0.02/−0.67pp; every load-level joint interval spans zero. Urgency improves the Medium sample point estimate by +2.27pp (pointwise95% [+0.07,+4.85]); supplies High changes−1.92pp with a wide [−8.22,+4.50] interval. The subtype result is exploratory among many comparisons and does not establish universal fairness protection. Coordination Low sample +0.80pp also has a nominal positive pointwise interval, without overall gain. All other subtypes and counts remain visible in the detailed report.

Tail uses819,200/921,600/1,024,000 transitions. Dev-extra is development. Only best models are evaluated on Test-v3, so last-model stability is not claimed on the holdout.

## Initial type diagnosis and remaining failures

Before training, frozen M5 and Gen2 destination policies were replayed over3,000 Dev-extra full days; every original day field and diagnostic remained exact. Sample changes at Medium/High were −0.69/−1.03 pp, with paired intervals[−2.36,+0.98]/[−3.19,+1.02] pp. These do not establish a persistent category sacrifice. No hard type tolerance was fitted to Test-v2 results. The sample High point decline primarily accompanies expired-unpicked +1.89pp, while expired-after-pick falls0.35pp and late falls0.51pp; it is not simply evidence of delivering picked samples too slowly.

Failure classification retains never-visible orders, first-visible/pre-pick direct slack, expired-before/after-pick outcomes, action underway at deadline crossings and urgent-cargo order-state counts. These are observational classifications, not proven avoidable errors. See [initial diagnosis](gen3_initial_diagnostics.md) and complete `failure_patterns` in the development/Test JSON. All four types and all loads are shown; no favorable subtype filtering.

## What changed

- **Urgency only:** the existing DELIVER head gains11 current-cargo risk features. Reconstruct minutes from type-normalized deadline/expiry values before minima; empty groups have zero minima and presence flags. Due/overdue ≤15min counts use half the shortest original30min window. Valid and expired physical cargo counts/weights remain explicit; other destinations’ urgent cargo is summarized. PICK, critic and other readout dimensions stay unchanged.
- **Coordination only:** PICK gains the same107-dimensional destination group already used by DELIVER. The visible pending summary includes the candidate itself and describes the current state; no counterfactual increments or hidden orders. DELIVER and other readouts retain the reference structure.
- No combined urgency+coordination model, reward/type weighting, wider candidate set, attention search or budget extension was performed.

## Equivalent inference and CPU cost

The reference already scores all six facilities in one batched call. The optional adapter reuses global context projections in PICK and DELIVER; near ties fall back to the canonical policy. Underlying parameter tensors are unchanged, and these adapters never enter PPO or the scientific final evaluator. Both factored and existing argmax-only modes were compared on complete Dev trajectories; all logits/value tolerances, actions and complete-day fields are retained in verification outputs.

| Method | Parameters | Canonical B1 p50/p95 (µs) | Argmax-only B1 | Factored+guard B1 |
|---|---:|---:|---:|---:|
| destination | 79,675 | 270.4/290.2 | 206.9/219.6 | 212.5/226.7 |
| urgency | 80,379 | 433.5/453.3 | 363.4/380.9 | 372.0/390.5 |
| coordinated | 86,523 | 298.5/316.6 | 231.4/242.1 | 240.9/256.7 |

| New method | Run wall-time range (s) | Sum of five run wall-times (s) |
|---|---:|---:|
| urgency | 1523.8–1541.3 | 7651.8 |
| coordinated | 1449.7–1484.2 | 7344.2 |

The existing argmax-only path is slightly faster at batch1 for this reference (206.9µs), while guarded factoring is faster at batch8 (306.7µs versus314.5µs argmax-only and384.7µs canonical). Factoring is not uniformly the fastest mode. The guarded path includes a canonical near-tie fallback; generic argmax-only has observed trajectory equivalence rather than a universal floating-point tie guarantee.

Five single-thread learners ran concurrently; some read-only development/inference checks overlapped training. These are observed run wall-times, not an isolated training-speed ratio. The suite manifest records actual elapsed time.

Single CPU thread, identical archived real observation/mask inputs, inference_mode, three variant-order-rotated rounds,200 warmups+2,000 timed calls per model/variant/batch. Table medians summarize per-model round-median quantiles, not pooled p95s. Batch8 and all raw round quantiles are in `gen3_inference.json`. Inference was measured after all project training/evaluation jobs stopped. Floating-point logits need not be bitwise identical; observed full-day behavior is checked, not assumed.

## Integrity and artifacts

- Ten full scratch runs,10,240,000 transitions,10,000 PPO updates and110 Dev checkpoints, source `8d5f1d3945f6e4086be4e108e227ee7d70d32776`.
- New Test-v3: 5,400 full episodes and 585,562 decisions; no illegal action or nonfinite required metric.
- [Training audit](gen3_training_audit.json), [development detail](gen3_development.md), [Test-v3 detail](gen3_test_v3.md), [learning curves](gen3_learning_curves.png), [tail curves](gen3_tail_curves.png), [paired Test-v3 differences](gen3_test_v3_paired.png).
- [Final integrity audit](gen3_final_audit.json) records preserved prior results/model hashes, frozen statistical code and all20-model inference trajectory checks.
- Protocols: `configs/gen3_holdout_freeze.json`, `configs/gen3_protocol.json`, `configs/gen3_test_v3_protocol.json`. Raw checkpoints, full-day and per-order failure records stay in the new `outputs/gen3_*` directories; these are ignored by Git and retained locally.
- No seed is chosen using Test-v3. Test-v3 is now consumed. No architecture/hyperparameter changes followed these results. Neither theoretical optimality nor new-map generalization is established.
