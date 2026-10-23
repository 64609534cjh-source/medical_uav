**I revise my earlier recommendation: use compromise-reward scratch PPO as the main pipeline.** The pilots justify that change; they do not yet establish set-network performance or robustness across training seeds. I inspected the scripts and logs, without rerunning experiments or editing files.

**A. Agree on compromise; qualify the claim.** Against on-time-only PPO, it gives medium −0.5 percentage points and high +0.1, while high completion improves 1.8 points and expiry falls 1.8 points. The planner screen supports the same trade-off. That is enough to choose it, but “equivalent on-time performance” needs multiple training seeds. Plasma is better than on-time-only in these pilots, though high-load PPO remains slightly below Slack-aware: .63 versus .64.

State the objective honestly:
\[
R=1.1N_{\mathrm{on}}+0.2N_{\mathrm{late}}-0.1N_{\mathrm{released}}-0.001E.
\]
This prioritizes timely service while rewarding late rescues. Freeze the coefficients; retain on-time-only as an ablation.

**B. Agree on scratch PPO; make the set network provisional.** Scratch PPO’s gains are substantial, while BC trails its teacher and DAgger gives inconsistent improvements. My previous insistence on imitation as the main pipeline is no longer warranted.

However, these are **MLP results**. Use the proposed set network as the main candidate, with the successful tanh MLP as a required matched control. Keep one uniformly mixed-load policy; the evidence removes the need for a low-load curriculum. Select MLP if three matched seeds show a clear advantage exceeding two points on the medium/high mean, or if set-network instability remains unresolved. Do not force an architectural success story.

**C. Agree.** Slack-aware is sufficient for M2’s demonstration that the network learns. Planner imitation belongs in M3 after the information audit. Keep both Slack-BC→PPO and planner-BC→PPO: comparing only planner initialization with scratch would leave the teacher’s contribution unexplained.

Do not schedule more DAgger by default. Its pilot does not justify making it a dependency.

**D. Agree.** Freeze the 452-feature observation, RETURN, charging commitments, 12 slots, and map-0 primary scope. Scenario rollout is the strongest planned conventional reference, not an assumed winner or an automatically “state-of-the-art” implementation.

One correction to my intermediate update: the pilot’s actual demand seeds **are** confined to `[0,100000)` by `reset`; I found no seed-range leakage. Production should still record actual demand seeds and separate demonstration and online-training pools.

**E. Agree with the owner’s deferral.** Reserve October 31–November 1 for visualization **if the owner approves that timing**. M2 must not depend on it. Preserve useful event logs now; later produce a short explanatory replay with held decision frames and a synchronized same-arrivals comparison.

**F. Agree.** One authoritative roadmap in `docs/`, containing the decision log and experiment ledger. Store larger configurations, raw results, and checkpoints separately and link them. Record schema/reward version, actual seeds, command, commit, checkpoint-selection rule, and artifact checksum—not just headline numbers.

The research question becomes: **Can scratch PPO learn better batching and charging decisions than rules and online planning, and how much do imitation, reward choice, and network structure contribute?**

The milestone plan is:

| Dates, 2026 | Deliverables and acceptance |
|---|---|
| **Oct 23–25** | Freeze reward and observation schema; implement both networks and the shared evaluator. Verify reward accounting, masking, permutation properties, and PPO trajectory boundaries. Record a versioned configuration. |
| **Oct 26–28** | Train Slack-aware BC; produce held-out learning curves and closed-loop results. Complete short scratch-PPO runs with both architectures. Implement FCFS/nearest and begin the planner information audit. |
| **Oct 29–30: M2** | Submit commented network/environment/BC code, trained BC checkpoints, forward/backward demo, lightweight evaluator, exact environment, README, and two-page report. Rehearse from the extracted archive on CPU. Keep unfinished PPO and exploratory scripts internal. Explain that BC does not train the value head; demonstrate its backward path separately if claimed. |
| **Oct 31–Nov 6: M3** | Complete three seeds of set-network scratch PPO and both BC→PPO comparisons, including frozen BC-only checkpoints. Deliver training/evaluation commands, learning curves, trained agents, conventional comparisons, and failure analysis. Follow the actual M3 rubric once available. Visualization work occupies the reserved slot only with owner agreement. |
| **Nov 7–11** | Complete architecture and reward controls; extend the selected primary architecture to five seeds. Validate and select scenario rollout. Finish optional map-transfer work only after core comparisons are complete. Freeze methods, checkpoint rules, and analysis before test evaluation. |
| **Nov 12–14** | Run locked paired tests; produce uncertainty estimates, per-type results, latency measurements, and batching/charging case studies. No performance tuning from test results. |
| **Nov 15–17** | Assemble final video, individual ≤10-page reports, reproducible package, and clean-machine rehearsal. Each member must explain the formulation and interpret a failure. |
| **Nov 18–20: final** | Packaging/correctness fixes and submission buffer. Ship training, validation/testing code, trained models, environment, README, video, and reports. Document any correctness fix requiring results to be regenerated. |

Use the pilot PPO settings as the starting configuration. Preserve their exact implementation in the ledger: for example, the pilot’s nested value-loss factors yield **0.25 × mean squared error**, which should not silently become 0.5 during cleanup.

The experiment budget should be explicit:

| Experiment | Budget |
|---|---:|
| Integration and targeted debugging | **≤3M decisions total** |
| Primary scratch PPO | **5 seeds × 6M = 30M** |
| Alternative-architecture scratch PPO | **3 × 6M = 18M** |
| Slack-BC→PPO | **3 × 6M = 18M** |
| Planner-BC→PPO | **3 × 6M = 18M** |
| Current-reward and on-time-only controls, using primary architecture | **2 × 3 × 6M = 36M** |
| Optional mixed-map scratch PPO | **3 × 6M = 18M additional** |

That is **123M core decisions**, including debugging. The architecture budgets cover either final choice: first run both for three seeds, then add seeds 44/55 to the selected architecture.

Additional budgets and controls:

- **BC:** 900 balanced episodes per teacher, at most 30 epochs; identical demonstration days across teachers. Split 80/20 by demand seed, grouping its occurrences across loads. M2 can show one initialization; complete three for the imitation comparisons.
- **Planner audit:** 100 saved states. Changing unavailable future arrivals or hidden-order details must leave candidate scores and choices unchanged, with public information held fixed.
- **Scenario rollout:** screen 4 versus 16 scenarios on 40 days × medium/high: **160 planner-days**. Use independent future-arrival RNG, common scenarios across candidate actions, and a fixed 60-minute arrival horizon. Select by development performance within a measured p95 one-second decision budget.
- **Imitation comparison:** retain 0-decision BC checkpoints and evaluate at 0.5M, 1M, 3M, and 6M PPO decisions. Report demonstration generation and pretraining costs separately; equal PPO budgets alone do not mean equal total cost.
- **No automatic training extension.** Six million already produces strong results. Any extension must be justified before test access and applied symmetrically to the affected comparison.

Budget from measured production throughput. The compromise pilot log took approximately **27 minutes for 6M decisions**, versus the earlier faster current-reward pilot. Set-network, planner-labeling, and evaluation costs still need measurement.

Use seeds **11/22/33** for controlled comparisons and add **44/55** for the primary agent. Keep **100000–100039** for screening, **100040–100139** for selection, and **200000–200299** locked for testing.

Select one checkpoint per training seed across all loads using the medium/high mean, retaining the earlier low-load ≥97% and per-type ≤3-point degradation gates. If none qualifies, report that failure. Report mean daily rates **and pooled counts**, completion, expiry, per-type denominators, energy, charging, batching, and latency. Bootstrap paired whole days; separately show training-seed variation. The existing intervals describe demand-day uncertainty for one trained model.

The **three most important things M2 week must do** are:

1. **Freeze and verify the experiment contract:** reward, 452-feature schema, masks, seed manifest, evaluator, and raw-result format.
2. **Demonstrate real network learning:** correct set/MLP behavior, grouped BC validation, closed-loop results, and an extracted-archive demo that works without training.
3. **Remove M3’s integration risk:** run production scratch PPO end to end on both networks, verify updates and episode boundaries, save/reload checkpoints, and measure throughput. Planner imitation and visualization must not block these essentials.