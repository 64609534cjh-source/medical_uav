Reviewed `d1d04fe..aa3e840` on `m3-ppo`; no files edited. I checked the saved data independently, including every selection eligibility flag, score and winner. Runtime tests were unavailable: this shell’s Python lacks PyTorch.

**A. Code review**

**Critical:** none established. I found no planner future-order leak, policy-changing warm-up, incorrect per-type pooling, or M1 reward mismatch.

**Important**

- **Batching metric counts unsuccessful unloads as delivered boxes.** [metrics.py:57](/home/caijihang/projects/medical_uav/medical_uav/metrics.py:57) counts every `unload` segment, including orders expiring during unloading. The simulator then disposes of those orders. Report successful `deliver` events per physical delivery stop separately from attempted unloads; otherwise failed service can inflate “better batching.”
- **Planner audit is insufficient for the roadmap claim.** [test_rollout.py:34](/home/caijihang/projects/medical_uav/tests/test_rollout.py:34) tests 20 states from five days, versus the roadmap’s 100. It never asserts that hidden pending orders exist, and only changes their destination/weight. Add explicitly overflowing queues and mutations of hidden deadlines, expiries and counts while holding visible slots fixed. The pruning implementation itself looks sound.
- **Reward/run identity does not travel with results and checkpoints.** [train_ppo.py:107](/home/caijihang/projects/medical_uav/scripts/train_ppo.py:107) omits reward variant, initialization and revision from checkpoint metadata; [evaluation.py:52](/home/caijihang/projects/medical_uav/medical_uav/evaluation.py:52) records only `v2` and demand seed. `run.json` helps, but exported checkpoints/CSVs can become ambiguous. Evaluation always uses v2: correct for common service comparisons, but its return is **v2-scored return**, including for M1-trained agents.
- **Latency needs a separate benchmark before reporting speed ratios.** [evaluate.py:49](/home/caijihang/projects/medical_uav/scripts/evaluate.py:49), [evaluation.py:44](/home/caijihang/projects/medical_uav/medical_uav/evaluation.py:44): parallel wall-clock timings include contention from workers/training. One thread per worker helps but does not remove it. Benchmark serially on an otherwise idle machine; retain decision samples for median/p95. Averaging daily p95s is not an overall p95.

**Minor**

- [ppo.py:157](/home/caijihang/projects/medical_uav/medical_uav/agents/ppo.py:157): warm-up “before/after” losses use different batches and changing GAE targets; “after” is measured **before the last optimization**. This does not demonstrate critic improvement. Use a fixed evaluation batch/target. Policy freezing is correct for both current architectures.
- [evaluation.py:104](/home/caijihang/projects/medical_uav/medical_uav/evaluation.py:104): NaNs pass the eligibility comparisons. Reject non-finite rates explicitly. Saved selections contain no observed failure here; padded checkpoint names also make the current tie ordering correct.
- [analyze_behaviour.py:57](/home/caijihang/projects/medical_uav/scripts/analyze_behaviour.py:57) drops `charge_share` despite the plan. Its hourly plot averages daily rates, rather than pooling counts, and plots only the last load. Label the estimand and show both loads with denominators.

**Checks that passed by inspection/data:** M1’s formula and coefficients match `m1-submission-v1`; warm-up freezes shared encoders/attention and actor parameters; selection uses full paths, pooled type rates, correct gates and `None` when none qualifies.

**B. What the results support**

From **final checkpoints on 40 screen days**, averaging the three set-network seeds:

| Metric | Medium: PPO / Slack | High: PPO / Slack |
|---|---:|---:|
| On-time | 91.71% / 80.99% | 83.04% / 65.89% |
| Served | 94.08% / 90.58% | 86.45% / 80.66% |
| Distance/day | 137.3 / 152.6 km | 138.8 / 156.1 km |
| Charging/day | 105.8 / 112.1 min | 107.8 / 116.6 min |

Day-paired bootstrap, averaging training seeds within each day: PPO–Slack **+10.72 pp [9.06, 12.39]** medium; **+17.15 pp [15.50, 18.77]** high. These intervals describe day uncertainty, not training-seed uncertainty.

The clearest explanation to investigate is **less wasted travel and late service**. Late-delivery share falls from 9.60%→2.37% medium and 14.76%→3.41% high, while total service rises.

| Hypothesis | Plausibility now | Measurement/plot that would support or weaken it |
|---|---|---|
| **Better batching/routing** | Strongest candidate: more service with ~10–11% less distance. | Reconstruct cargo from pickup/deliver/dispose events; plot successful boxes/stop, flight-distance-weighted occupancy, empty-flight share, km/on-time delivery. Higher occupancy/multi-box stops supports batching; unchanged batching with fewer empty legs instead supports routing/repositioning. |
| **Earlier/smarter charging** | Plausible; fewer charging minutes alone cannot establish timing. | From charge segments: start battery, target, station, duration, peak overlap; reconstruct pending feasible orders and cargo at charge start. Plot charging against demand windows and deadline misses during charging. Support requires less charging interference with urgent work, not merely higher start battery. |
| **Abandoning hopeless work** | Plausible selective avoidance; indiscriminate abandonment is inconsistent with higher served rate. | At each PICK, reconstruct pre-action location/time and direct-completion slack. Plot pickups already impossible to deliver on time, travel/handling spent on them, and pending-versus-onboard expiries. Fewer futile pickups alongside more timely deliveries supports this mechanism. |
| **Pre-positioning before peaks** | Unproven. | Align traces to peak starts at minutes 60 and 300: location, battery, empty RETURNs, charging completion during the preceding 30 minutes; then pickup delays during the first 30 peak minutes. Support requires anticipatory positioning before backlog rises, followed by faster response. |
| **Better plasma handling** | Supported as an outcome; mechanism unknown. | Plot pooled plasma on-time/late/expired counts across checkpoints, plus release→pickup and pickup→delivery distributions, charge overlap while carrying plasma, and detours after pickup. Include unpicked expiries so faster surviving deliveries cannot conceal selection bias. |

Compute these on matched days for **all selected seeds**, Slack and the audited planner, separately by load. Use paired day differences and show seed spread. Aggregate associations explain candidates; same-state action substitutions with a common continuation would provide stronger evidence that a particular decision causes the gain.

**What the two-page M3 report should say**

“Across three training seeds on map-0 development days, scratch PPO substantially improves timely service over Slack-aware, with fewer expiries and less travel. The set network modestly outperforms the MLP. Checkpoint selection explicitly protects low-load service and per-type performance.”

Use one performance table, one learning-curve panel and one matched behavioural example backed by aggregate measurements.

Do **not yet claim** confirmed test performance, causal batching/charging explanations, superiority to the production planner, benefit from BC initialization, reward superiority at convergence, or generalization to other maps.

**Checks before reporting**

- **Checkpoint mismatch:** screen results use all 6M finals; selected set checkpoints are **6M / 5.5M / 4.5M**. Re-evaluate those selected models before presenting a “selected agent” table. Selection-day scores are selection evidence, not independent validation.
- **Old CSV schema:** `m3_screen_eval.csv` lacks per-type counts and latency. Its plasma means are daily averages; do not present them as pooled proportions or compare directly with pooled selection rates.
- **Reward controls are early:** at the common ~0.5M point, three-seed medium/high means are v2 **.883/.770**, M1 **.865/.716**, on-time-only **.890/.778**. This suggests slower early learning under M1, not its final inferiority.
- **Plasma population shift:** Slack’s high-load plasma score is about .636 in the screen daily averages versus .594 pooled on selection days. Recompute consistently before interpreting the difference.
- **Roadmap §9.6:** “same days” Slack numbers `.810/.659` match the screen baseline, while that paragraph describes the first 20 selection days. Verify the baseline provenance.
- The production planner result and BC→PPO comparison remain outstanding; pilot planner numbers should stay explicitly labelled as pilots.