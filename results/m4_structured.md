# M4: shared-order encoder vs frozen flat MLP

All results below use the same reused Dev demand days. Test remains unused. Best checkpoints were independently selected by the frozen equal-load on-time rule, with daily reward as the tie-breaker. No learner seed was selected or dropped.

## Selected checkpoints

| Seed | Best step | Low % | Medium % | High % | Equal % | MLP equal % | Paired change (pp) | Last equal % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 614,400 | 99.39 | 85.43 | 75.11 | 86.64 | 77.66 | +8.98 | 86.31 |
| 2 | 512,000 | 100.00 | 86.46 | 72.80 | 86.42 | 78.40 | +8.02 | 84.48 |
| 3 | 819,200 | 99.61 | 85.20 | 74.60 | 86.47 | 78.35 | +8.12 | 85.26 |
| 4 | 716,800 | 99.90 | 86.40 | 74.51 | 86.94 | 78.18 | +8.76 | 82.80 |
| 5 | 716,800 | 99.53 | 85.28 | 74.47 | 86.43 | 77.26 | +9.17 | 85.14 |

## Across five learner seeds

Mean ± sample SD (ddof=1); these are not confidence intervals. Matching seed numbers organize repetitions but do not imply identical trajectories across architectures.

| Model / selection | Low % | Medium % | High % | Equal % |
|---|---:|---:|---:|---:|
| MLP best | 98.68 ± 0.39 | 74.31 ± 0.97 | 60.93 ± 1.33 | 77.97 ± 0.49 |
| Structured best | 99.69 ± 0.26 | 85.75 ± 0.62 | 74.30 ± 0.88 | 86.58 ± 0.22 |
| MLP last | 98.52 ± 0.51 | 73.68 ± 1.25 | 59.44 ± 1.48 | 77.22 ± 0.98 |
| Structured last | 99.68 ± 0.04 | 83.61 ± 1.41 | 71.11 ± 2.67 | 84.80 ± 1.29 |
| slack | 99.26 | 80.99 | 65.89 | 82.05 |
| edf | 99.56 | 40.56 | 27.47 | 55.86 |
| random | 51.14 | 23.55 | 21.28 | 31.99 |

| Paired learner change, Structured − MLP | Low (pp) | Medium (pp) | High (pp) | Equal (pp) |
|---|---:|---:|---:|---:|
| best | +1.01 ± 0.49 | +11.45 ± 1.01 | +13.37 ± 2.00 | +8.61 ± 0.51 |
| last | +1.16 ± 0.49 | +9.92 ± 0.97 | +11.67 ± 2.37 | +7.58 ± 1.01 |

## Training trajectory

The pointwise equal-load mean changes from 82.07% at 102,400 steps to 84.80% at 1,024,000 steps (+2.73 pp). The last three checkpoint means are 85.71% / 85.15% / 84.80%.
Final-interval changes for seeds 1–5 (pp): +1.36 / -1.44 / +0.34 / -1.85 / -0.15.
Seeds with best before last: [1, 2, 3, 4, 5]. These finite-budget trajectories do not establish convergence.

[Full curves](m4_structured_curves.png) · [Post-100k curves](m4_structured_tail_curves.png) · [Matched-day differences](m4_structured_paired.png)

Each change uses the corresponding learner seed's selected checkpoint. The complete 1,200 best/last × seed × load × day differences, including reward changes and Slack comparisons, are in `m4_structured_paired.json`. Day-level counts are descriptive; the 600 best day pairs are not 600 independent training repetitions.

## Matched-day differences for selected best models

| Load | Mean change (pp) | Win / tie / loss across 200 seed-day pairs | Days all five win | Days all five lose |
|---|---:|---:|---:|---:|
| low | +1.01 | 40 / 154 / 6 | 0 | 0 |
| medium | +11.45 | 185 / 8 / 7 | 30 | 0 |
| high | +13.37 | 196 / 1 / 3 | 36 | 0 |

## Service and training objective

Each cell is a mean per complete demand day, averaged equally across learner seeds. Baseline rows use the identical days.

| Load / model | On time % | Reward | On-time count | Late count | Expired | Flight energy |
|---|---:|---:|---:|---:|---:|---:|
| low / slack | 99.26 | 21.143 | 21.400 | 0.175 | 0.000 | 355.21 |
| low / mlp_best | 98.68 | 20.771 | 21.240 | 0.270 | 0.065 | 603.27 |
| low / structured_best | 99.69 | 20.972 | 21.495 | 0.070 | 0.010 | 568.38 |
| low / structured_last | 99.68 | 21.110 | 21.495 | 0.075 | 0.005 | 441.95 |
| medium / slack | 80.99 | 40.434 | 41.550 | 5.075 | 5.200 | 604.77 |
| medium / mlp_best | 74.31 | 38.447 | 38.000 | 7.900 | 5.925 | 664.48 |
| medium / structured_best | 85.75 | 45.132 | 44.150 | 5.130 | 2.545 | 588.16 |
| medium / structured_last | 83.61 | 45.497 | 42.965 | 6.980 | 1.880 | 587.41 |
| high / slack | 65.89 | 40.551 | 44.025 | 10.125 | 13.750 | 628.54 |
| high / mlp_best | 60.93 | 37.728 | 40.580 | 11.170 | 16.150 | 677.50 |
| high / structured_best | 74.30 | 50.374 | 49.845 | 10.260 | 7.795 | 577.23 |
| high / structured_last | 71.11 | 50.805 | 47.635 | 13.355 | 6.910 | 573.74 |

| Load / order type | MLP best on time % | Structured best on time % | Slack on time % |
|---|---:|---:|---:|
| medium / plasma | 65.52 | 68.89 | 74.25 |
| medium / medicine | 87.63 | 94.18 | 87.90 |
| medium / sample | 65.22 | 90.26 | 80.07 |
| medium / supplies | 75.53 | 84.07 | 79.15 |
| high / plasma | 57.46 | 56.65 | 63.59 |
| high / medicine | 74.33 | 82.77 | 73.56 |
| high / sample | 52.64 | 79.61 | 65.56 |
| high / supplies | 58.66 | 71.13 | 59.47 |

Per-type rates average complete demand days with at least one order of that type; denominators and other service outcomes are in the JSON.

## M4-A follow-up: seed 4 Dev best vs last

The preceding-update windows are in `m4a_diagnostics.md`. This table completes the daily service/action-time comparison; action durations include flight and handling within the macro-action.

| Load / selection | On time % | Reward | Expiry % | PICK min | DELIVER min | CHARGE min | WAIT min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low / best | 98.16 | 20.539 | 0.47 | 80.21 | 90.30 | 203.55 | 60.14 |
| low / last | 97.70 | 20.608 | 0.21 | 83.92 | 90.10 | 201.18 | 66.07 |
| medium / best | 75.74 | 37.788 | 12.24 | 152.26 | 183.54 | 134.96 | 33.75 |
| medium / last | 72.08 | 38.245 | 10.27 | 163.06 | 170.40 | 139.93 | 34.51 |
| high / best | 60.63 | 36.856 | 23.51 | 165.83 | 203.45 | 132.60 | 24.58 |
| high / last | 57.78 | 37.978 | 21.75 | 179.47 | 190.75 | 134.75 | 24.73 |

Seed 4's lower final on-time rate accompanies higher reward and lower expiry in Medium/High. The recorded windows show no KL/value explosion. This remains an observed objective/service tradeoff; the logs do not identify a unique causal explanation.

Per-type daily on-time, delivered, late and expiry rates (with empty-type days excluded and denominators recorded) are in the summary JSON. Action time distributions and per-seed best/last changes are retained there and in the audit JSON.

## Behavior on Medium and High

All selected structured best/last models are replayed on all 80 Medium/High days and checked against their saved reward and on-time records. The M4-A replay supplies the same-day MLP and Slack references. Means below are event-weighted; numerators and denominators are retained.

| Load / model | Boxes/drop | Batch drops % | Picked then expired % | Charge arrival battery | Charge target | Hidden orders/state |
|---|---:|---:|---:|---:|---:|---:|
| medium / slack | 1.270 | 22.34 | 1.95 | 28.69 | 80.14 | 0.046 |
| medium / mlp_best | 1.160 | 12.83 | 1.64 | 36.03 | 68.66 | 0.070 |
| medium / structured_best | 1.320 | 24.29 | 0.53 | 55.45 | 83.53 | 0.008 |
| medium / structured_last | 1.275 | 21.73 | 0.54 | 57.08 | 84.83 | 0.012 |
| high / slack | 1.413 | 32.88 | 3.17 | 26.42 | 80.14 | 0.657 |
| high / mlp_best | 1.161 | 13.62 | 2.60 | 31.81 | 67.53 | 0.796 |
| high / structured_best | 1.452 | 32.57 | 0.89 | 49.73 | 82.13 | 0.232 |
| high / structured_last | 1.425 | 31.20 | 0.89 | 46.66 | 80.98 | 0.260 |

Compared with MLP best, the structured best models show higher batching and lower expiry after pickup on both loads. Charging occurs at higher arrival battery and with higher target levels, while mean daily flight energy is lower. The joint service/energy results support the narrower observation that this learned scheduling behavior differs in useful ways; charging frequency alone is not a causal explanation.

These policies visit different states. Batching, charging and expiry differences are associations, not proof of the mechanism causing any score change. Pickup slack, payload, demand at charging, waiting with legal service, legal counts and normalized entropy remain in the full aggregates and raw behavior days.

## Compute and reproducibility

| Architecture | Parameters | Masked argmax batch 1 (µs) | Batch 8 (µs) | Batch 256 (µs) |
|---|---:|---:|---:|---:|
| mlp | 139,803 | 93.38 | 128.33 | 893.11 |
| structured | 65,232 | 199.96 | 292.85 | 4378.16 |

Inference uses one CPU thread, the same 256 archived real observations, 50 warmups, seven repetitions of 300 calls, with no experiment learner running. The structured model has fewer parameters but more shared per-order computation and higher inference latency.

| Seed | Complete training days | Training stages (s) | Actual training transitions/s | Run wall (s) | Process CPU user+system (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,203 | 872.37 | 1173.8 | 996.61 | 997.89 |
| 2 | 9,432 | 870.87 | 1175.8 | 993.00 | 994.50 |
| 3 | 9,245 | 865.53 | 1183.1 | 989.60 | 991.06 |
| 4 | 9,106 | 868.20 | 1179.5 | 995.50 | 997.00 |
| 5 | 9,295 | 872.78 | 1173.3 | 996.26 | 997.69 |

M4 formal runs used five independent local CPU processes, one torch thread each. Orchestrator elapsed time was 998.13 s; summed process wall time was 4970.96 s. M3 was serial, so elapsed time is not a direct compute comparison. The M4 seed-0 pilot used 42.52 s for training stages (2408.2 transitions/s), versus the earlier MLP pilot's 21.99 s (4655.9 transitions/s); these were separate one-thread pilot measurements, not a controlled simultaneous timing trial.

All 5,000 updates and 55 full-Dev checkpoints passed the saved-source, budget, selection, finite-tensor, legal-action and complete-day audits. There were 46,281 complete training episodes and 6,600 PPO Dev episodes. All five initial parameter digests differ. M3 models, logs and selection rules were preserved. Training source commit: `8a5a76a8407bac3ed66e2169ab5361324733512e`. Frozen protocol: `configs/m4_protocol.json`.

## Interpretation

All five corresponding learner comparisons improve (+8.02 to +9.17 pp). All best checkpoints occur before the final step. The final mean is -1.78 pp below selected best, while equal-load mean daily reward changes by +0.311. Medium/High expiry is lower at last despite lower on-time rate. This observed tradeoff does not establish convergence or a unique causal explanation for the decline.
The selected structured architecture changes equal-load Dev on-time rate by +8.61 ± 0.51 pp across the five corresponding learner seeds, and its mean is +4.53 pp relative to Slack. This is evidence for this specific architecture and training protocol on reused Dev; it neither isolates parameter count from structural bias nor establishes held-out generalization.

No network, reward or PPO tuning followed the pilot. No Medium-only, BC, larger-slot or memory-network experiment was mixed into M4. Test remains sealed until candidate methods and the final comparison protocol are frozen.
