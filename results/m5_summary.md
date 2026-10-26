# M5: bounded stability tuning, inference reuse and frozen Test

M4 S0 was preserved. Only the preregistered S1 late-learning-rate schedule was trained; no architecture/reward/entropy/slot change or combination search was added. S2 was not run because the pre-tuning diagnosis did not motivate stricter update control.

## Independent development-day diagnosis

Dev-extra 100040–100139 was unused in recorded local experiments before M5. It is now explicitly part of development, not Test. Existing M4 best/last, MLP best/last and Slack were evaluated before tuning. M4 best is 85.96%, MLP best 77.45%, Slack 81.47%; M4 last is 84.51%. All five M4 best-to-last declines repeat on the extra days. These results support investigating a policy-level tradeoff, while the original best/main gap also contains selection optimism.

The [initial diagnostic](m5_initial_diagnostics.md) retains per-seed/load/type rates and counts, unpicked/picked expiry, effective-drop denominators, transport/resource measurements, LR and KL stopping. M4 main results were exactly reproduced.

## Single-factor schedule test

S1 keeps the actual M4 constant LR 3e-4 through update 500, then decays linearly to 3e-5 at update 1000. Every run starts from scratch and completes 1,024,000 transitions with the same 11 Dev-main checkpoints and same selection rule. Midpoint model tensors and collector sampler/counters exactly reproduce M4; the golden updater check shows that added diagnostics leave original updates unchanged. Exact categorical KL and legal-count-normalized entropy are now logged, without changing joint KL stopping.

| Development gate (seeds 1–3) | Observed change (pp) | Required change (pp) | Pass |
|---|---:|---:|---|
| best_main | +0.28 | ≥ -0.30 | True |
| last_main | +0.70 | ≥ +0.30 | True |
| tail_main | +0.35 | ≥ +0.30 | True |
| best_extra | +0.16 | ≥ -0.30 | True |
| last_extra | +0.63 | ≥ +0.00 | True |
| minimum per-load extra best | +0.11 | ≥ −2.00 | True |
| minimum per-type extra best | -2.33 | ≥ −3.00 | True |

The complete preregistered gate passed, so scratch seeds 4–5 were also completed.
These are fixed practical selection tolerances, not significance tests or statistical noninferiority claims. The final method was selected before Test.

| S1 seed | Best step | Best % | Last % | Tail % | Main best change (pp) | Main last change (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024,000 | 86.87 | 86.87 | 85.97 | +0.23 | +0.56 |
| 2 | 512,000 | 86.42 | 85.38 | 85.46 | +0.00 | +0.89 |
| 3 | 921,600 | 87.06 | 85.90 | 86.53 | +0.59 | +0.64 |
| 4 | 1,024,000 | 86.92 | 86.92 | 86.57 | -0.02 | +4.12 |
| 5 | 1,024,000 | 86.31 | 86.31 | 86.15 | -0.12 | +1.17 |

| Extra best service tradeoff (pp) | Plasma | Medicine | Sample | Supplies |
|---|---:|---:|---:|---:|
| low | +1.15 | +0.02 | -0.14 | +0.34 |
| medium | +4.03 | -0.35 | -1.32 | -0.37 |
| high | +3.97 | -0.95 | -1.74 | -1.29 |

These are on-time daily-rate changes on the explicitly developmental extra set, using the same selected best checkpoint per learner. Overall stability improvement does not imply that every order type improves; the supplies tradeoff is retained rather than hidden by the equal-load mean.

Each tail is the arithmetic mean of existing checkpoints 819200, 921600 and 1024000; it creates no extra selection opportunities. Per-load/type extra comparisons, actual LR, exact/approximate KL, normalized entropy, optimizer steps, early-stop counts and complete source/checkpoint audits remain in the JSON.

## Inference-only engineering change

The training network is unchanged. A separate inference subclass reuses the original PICK layer weights as Wc·context + We·order + bias; context projection is computed once. An action-only path bypasses categorical log-probability/entropy work. Near legal-logit ties (gap ≤1e-5), it falls back to the original categorical decision. Value outputs and state_dict keys remain compatible; float PICK logits are checked numerically, not claimed bit-identical.

| Path | B=1 p50 / p95 (µs) | B=8 p50 / p95 (µs per batch) | B=8 states/s at p50 |
|---|---:|---:|---:|
| mlp_original | 92.71 / 106.63 | 134.50 / 147.96 | 59478 |
| structured_original | 201.87 / 216.11 | 304.57 / 322.92 | 26266 |
| structured_argmax_only | 141.19 / 146.42 | 228.47 / 238.51 | 35015 |
| structured_factored_argmax | 149.62 / 156.20 | 226.05 / 236.75 | 35391 |

Full Dev (120 days, 12,721 decisions): original median 4.434 s, factored argmax median 3.694 s across three repetitions. One CPU thread, identical archived real observations, 200 warmups and 2,000 individually timed calls per microbenchmark. No learner or other evaluation suite was active.
Equivalence was checked over 1,800 complete Dev days across 15 preselected checkpoints: every action and every complete-day record field reproduced the canonical results. Maximum measured logit error: 3.81e-06; value error: 0.
Linear MAC estimates are 139,392 (MLP), 267,584 (original structured), 177,472 (factored structured); 90,112 repeated MACs are removed. These are analytic operation counts, not measured latency or memory. On B=1, argmax-only is slightly faster than the factored variant; extra small-tensor calls offset the saved MACs. B=8 is similar. The main latency reduction comes from the complete action-only inference path, and no proportional MAC-to-latency speedup is claimed. The scientific Test suite uses the canonical evaluator, so engineering and learning changes remain distinct.

## Frozen final Test

Final selected structural method: **S1**. All models, checkpoint hashes, comparisons, metrics and the paired learner/day bootstrap were committed before any Test evaluation.

| Method | Test equal on-time % | Learner SD (pp) | Joint 95% interval % |
|---|---:|---:|---:|
| mlp | 77.72 | 0.76 | [76.51, 78.89] |
| m4 | 86.22 | 0.32 | [85.46, 86.98] |
| s1 | 86.42 | 0.37 | [85.63, 87.20] |
| random | 32.33 | — | [31.18, 33.54] |
| edf | 56.09 | — | [54.60, 57.71] |
| slack | 82.12 | — | [81.22, 83.04] |

The original same-PPO architecture advantage repeats on Test: M4 − MLP is +8.50 pp. Final S1 − Slack is +4.30 pp, joint 95% interval [+3.77, +4.83] pp.
S1 − M4 on Test is +0.20 pp, with interval [-0.17, +0.61] pp. It includes zero: there is no clear additional selected-checkpoint performance gain. The demonstrated M5 benefit is the predeclared main/extra development-set last/tail stability improvement, while retaining the architecture's Test advantage.
Low does not show a clear advantage over Slack (-0.16 pp; interval [-0.45, +0.14] pp). Medium and High provide the positive final aggregate gain. No claim is made that every load or order type improves.

The [final Test report](final_test.md) includes all learner/load/type results and paired comparison intervals, with seed-only, day-only and joint uncertainty retained separately. The 100 demand-day IDs are resampled as common blocks across all loads/policies, and five learner repetitions retain their own level. No 1,500-independent-training-run assumption is made.

M3→M4 remains the original same-PPO architecture comparison. M4→M5 is a separately budgeted, explicitly limited tuning experiment. A failed or successful S1 result does not alter the original M4 claim. Test is now consumed; no model or schedule was revised after observing it. Fixed-map demand-day generalization does not establish new-map or real-world performance.
