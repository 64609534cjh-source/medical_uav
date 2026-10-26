# R1: development failure audit

observational Dev-extra; five learner repetitions of 100 shared days; no causal attribution

Outcomes are mutually exclusive. Rates below average eligible day/type rates, then learners; zero-release type days are omitted. Counts repeat each day under five policies.

| Method | Load | Type | Released* | Eligible days | On-time % | Late % | Expired unpicked % | Expired after pick % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| m5 | low | plasma | 2285 | 100 | 98.44 | 1.53 | 0.00 | 0.03 |
| m5 | low | medicine | 3405 | 100 | 100.00 | 0.00 | 0.00 | 0.00 |
| m5 | low | sample | 3340 | 100 | 99.81 | 0.19 | 0.00 | 0.00 |
| m5 | low | supplies | 2100 | 98 | 99.90 | 0.10 | 0.00 | 0.00 |
| m5 | medium | plasma | 5515 | 100 | 72.97 | 16.69 | 9.24 | 1.10 |
| m5 | medium | medicine | 7390 | 100 | 93.52 | 5.95 | 0.47 | 0.06 |
| m5 | medium | sample | 7725 | 100 | 88.52 | 9.43 | 1.72 | 0.32 |
| m5 | medium | supplies | 5340 | 100 | 82.85 | 13.84 | 3.05 | 0.27 |
| m5 | high | plasma | 6965 | 100 | 58.28 | 18.40 | 22.27 | 1.04 |
| m5 | high | medicine | 9985 | 100 | 82.44 | 14.05 | 3.11 | 0.39 |
| m5 | high | sample | 10160 | 100 | 76.96 | 15.01 | 7.24 | 0.78 |
| m5 | high | supplies | 6820 | 100 | 69.61 | 20.27 | 9.27 | 0.85 |
| destination | low | plasma | 2285 | 100 | 98.91 | 1.04 | 0.05 | 0.00 |
| destination | low | medicine | 3405 | 100 | 99.86 | 0.14 | 0.00 | 0.00 |
| destination | low | sample | 3340 | 100 | 99.60 | 0.40 | 0.00 | 0.00 |
| destination | low | supplies | 2100 | 98 | 99.35 | 0.65 | 0.00 | 0.00 |
| destination | medium | plasma | 5515 | 100 | 75.51 | 13.87 | 10.03 | 0.59 |
| destination | medium | medicine | 7390 | 100 | 93.84 | 5.66 | 0.44 | 0.06 |
| destination | medium | sample | 7725 | 100 | 87.83 | 9.80 | 2.28 | 0.08 |
| destination | medium | supplies | 5340 | 100 | 86.47 | 11.27 | 2.10 | 0.16 |
| destination | high | plasma | 6965 | 100 | 59.56 | 16.26 | 23.66 | 0.52 |
| destination | high | medicine | 9985 | 100 | 83.33 | 12.97 | 3.43 | 0.27 |
| destination | high | sample | 10160 | 100 | 75.94 | 14.50 | 9.13 | 0.43 |
| destination | high | supplies | 6820 | 100 | 74.84 | 15.83 | 9.05 | 0.28 |

*Released counts include five repetitions of the same demand days.

| Method | Load | Over-12 decision states (%) | Hidden direct-on-time matching states (%) | Never-visible failures / all failures (%) |
|---|---|---:|---:|---:|
| m5 | low | 0.000 | 0.000 | 0.000 |
| m5 | medium | 0.536 | 0.104 | 0.000 |
| m5 | high | 6.630 | 1.889 | 0.000 |
| destination | low | 0.000 | 0.000 | 0.000 |
| destination | medium | 0.545 | 0.130 | 0.000 |
| destination | high | 6.881 | 2.083 | 0.000 |

## Representative days

These are descriptive cases selected from Dev by the fixed bucket rule, not effect-size evidence. Full paired decision timelines are in `discussion_traces.json.gz`, reproduced by `scripts/trace_discussion.py`.

| Bucket | Learner | Load | Day | Destination % | M5 % | Delta pp |
|---|---:|---|---:|---:|---:|---:|
| success | 1 | low | 100088 | 100.00 | 100.00 | +0.00 |
| failure_both | 1 | medium | 100104 | 84.21 | 80.70 | +3.51 |
| improvement | 1 | low | 100073 | 100.00 | 96.30 | +3.70 |
| regression | 1 | high | 100117 | 67.11 | 71.05 | -3.95 |

## Interpretation and next tests

Candidate truncation is a testable interface hypothesis, not an established cause: report its actual prevalence above before prioritizing K=24. Never visible at decision times also includes orders that arrive and expire within a macro action; it is not synonymous with top-K exclusion.

Short-horizon planning is a separate online-compute hypothesis. Action-underway deadline crossings cannot determine avoidability. Any planner must reconstruct only public state and sample independent future scenarios; diagnostic simulator details cannot enter its policy input.

Prioritize a candidate-window feasibility/control test and a public-information lookahead feasibility test after these descriptive checks. Training settings, reward, type guardrails and checkpoint selection remain unchanged. R1 is partial until representative traces and new development registrations are reviewed; no new split is asserted unused.
