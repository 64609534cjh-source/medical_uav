# Gen3 pre-training: all-type failure diagnosis (Dev-extra only)

Frozen M5 and Gen2 destination policies were replayed on the same 100 Dev-extra days × three loads × five learners. After removing the new failure instrumentation, every original day field and diagnostic is exactly unchanged. No original Test or Test-v2 was rerun.

| Type | Load | Eligible days | Destination − M5 on-time (pp, paired 95%) | Late delta | Expired unpicked delta | Expired after pick delta |
|---|---|---:|---|---:|---:|---:|
| plasma | low | 100 | +0.46 [-0.52, +1.43] | -0.49 | +0.05 | -0.03 |
| plasma | medium | 100 | +2.54 [-1.22, +5.99] | -2.82 | +0.79 | -0.51 |
| plasma | high | 100 | +1.28 [-0.85, +3.46] | -2.15 | +1.39 | -0.53 |
| medicine | low | 100 | -0.14 [-0.50, +0.00] | +0.14 | +0.00 | +0.00 |
| medicine | medium | 100 | +0.32 [-0.78, +1.44] | -0.29 | -0.03 | -0.00 |
| medicine | high | 100 | +0.89 [-1.37, +3.25] | -1.08 | +0.31 | -0.12 |
| sample | low | 100 | -0.22 [-0.83, +0.21] | +0.22 | +0.00 | +0.00 |
| sample | medium | 100 | -0.69 [-2.36, +0.98] | +0.37 | +0.56 | -0.24 |
| sample | high | 100 | -1.03 [-3.19, +1.02] | -0.51 | +1.89 | -0.35 |
| supplies | low | 98 | -0.55 [-1.48, +0.00] | +0.55 | +0.00 | +0.00 |
| supplies | medium | 100 | +3.63 [+0.06, +7.66] | -2.57 | -0.95 | -0.11 |
| supplies | high | 100 | +5.23 [-0.20, +10.92] | -4.44 | -0.22 | -0.57 |

All outcome deltas are mean daily type rates; zero-release days are omitted consistently. On-time + late + expired-unpicked + expired-after-pick differences sum to zero exactly. Interval resampling pairs learner/day IDs, sharing day weights across loads and renormalizing weights over eligible days. Counts, separate learner/day uncertainty, all outcomes and per-order failure details are retained in JSON/raw outputs.

Deadline crossings during PICK/DELIVER/CHARGE/WAIT/RETURN describe the action underway, not an avoidable causal error. Direct slack ignores alternative routes, capacity and energy. First-visible status and never-visible failures are diagnostics only; hidden simulator order details do not enter model inputs. Counts aggregate five repetitions of the same demand days, not independent demand samples.

The original destination summary contains nonlinear mean cargo/order embeddings and counts, not explicit physical-time minima. This motivates the fixed urgency statistics without claiming that learned embeddings lose all urgency information. The coordination experiment separately tests whether PICK benefits from the existing destination context. No reward weights, per-type tolerance or model-selection guardrail is introduced.
