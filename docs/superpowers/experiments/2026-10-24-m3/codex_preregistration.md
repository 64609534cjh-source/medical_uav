# Codex: warm-up review and pre-registered analysis of the control runs (2026-10-26)

Written before `select_checkpoints.py` ran on the control runs. Already seen at that time: the 20-selection-day training curves (`evals.csv`) of the reward-control runs and the first evaluations of the BC → PPO and seed 44/55 runs; no 100-day selection or screen-day result of any control run.

Robustness points in Part A (empty held-out batch, undefined explained variance) are not handled: with 512 decisions per environment and days of 80–120 decisions, every environment finishes several days.

1. **Part A: warm-up is correct for the intended runs.** `day_returns` includes each terminal reward, resets accumulation at every terminal transition, and excludes each environment’s unfinished tail. A rollout starting mid-day still gives valid return-to-go for its observed states. These are exact **realized, reward-scaled returns**, not exact expected values. The diagnostic batch never enters optimization; subsequent collection advances beyond it. Both architectures keep actor parameters and shared encoders frozen, with no dropout/BatchNorm effects. Training still uses GAE targets; only the diagnostic uses Monte Carlo returns.

2. **Minor robustness bugs and documentation.**
   - [ppo.py:153](/home/caijihang/projects/medical_uav/medical_uav/agents/ppo.py:153): if **no environment finishes**, concatenating empty lists crashes. Handle an empty diagnostic explicitly.
   - [ppo.py:175](/home/caijihang/projects/medical_uav/medical_uav/agents/ppo.py:175): zero return variance—or fewer than two observations—makes explained variance undefined. Report it as unavailable.
   
   **40 rollouts is defensible**, given the reported EV improvement, but is a diagnostic-informed engineering choice, not an optimum. Its rationale is documented in roadmap §10 by subsequent commit `c47ed7a`; [Task 5:81](/home/caijihang/projects/medical_uav/docs/superpowers/plans/2026-10-24-m3-agent.md:81) remains stale. Freeze 40 across all three runs. The diagnostic is held out from fitting, but informed this budget choice. Runtime tests were unavailable because this shell lacks PyTorch.

3. **Part B: freeze the common analysis now.** Use matched seeds **11/22/33**, each independently selected on all 100 selection days under §4. On the 40 screen days, primary score \(S\) is the equally weighted medium/high **mean daily on-time rate**. Average paired variant differences across seeds within each day; bootstrap whole day IDs jointly across seeds and loads, **10,000 resamples, 95% CI**. Report each load and all three seed differences separately; these CIs measure day uncertainty, not training-seed uncertainty. Missing eligible checkpoints are selection failures, never silently dropped or replaced by finals. These remain development results: screen days and earlier evidence were already examined.

4. **Reward comparisons: practical threshold = 1 percentage point (pp).**
   
   For a directional claim, support requires mean gain **≥1 pp**, CI lower bound **>0**, and positive differences in at least **2/3 seeds**. A CI upper bound **<1 pp** rules out the stipulated practical gain; an upper bound **<0** supports reversal. Otherwise, inconclusive. A CI wholly inside **[−1,+1] pp** supports practical similarity.
   
   | Comparison | Registered direction | Permitted M3 wording |
   |---|---|---|
   | **m1 vs v2** | \(\Delta=S_{v2}-S_{m1}\) | Support: “v2 improves timely service under the fixed budget and selection protocol.” Refuted: “The predicted practical benefit was not reproduced,” or report reversal. Otherwise: “Uncertain.” This tests the **whole reward change**, including expiry, not late reward alone. |
   | **ontime vs v2** | \(\Delta=S_{ontime}-S_{v2}\) | Support: “Optimizing on-time alone improves the aligned metric.” Similarity: “Comparable timely service.” Otherwise report uncertainty or v2’s advantage. Alignment is by construction; superiority would not establish a universally better objective. |
   
   For the compromise rationale, pre-register secondary **v2−ontime served-rate** and **high-load pooled plasma on-time** differences. Claim a service trade-off only if on-time noninferiority holds (lower CI for \(S_{v2}-S_{ontime}>-1\) pp) and the claimed secondary gain is ≥1 pp with CI above zero. Report both secondary outcomes regardless.

5. **BC→PPO vs scratch: separate endpoint quality from efficiency.**
   - **Endpoint:** primary \(\Delta=S_{BC}-S_{scratch}\), using the same 1-pp rule above. Support permits “BC initialization plus critic warm-up improves selected performance”; equivalence permits “similar selected performance.” Neither establishes efficiency or isolates BC from warm-up.
   - **Efficiency:** on each seed’s 20-day curves, fix target \(T_s=0.95S_{scratch,s}(6M)\). Measure the first logged crossing sustained at the next evaluation; no interpolation, and no confirmed crossing means censored. Primary efficiency statistic: median paired decision saving. Call it practically faster only with **≥0.5M total-cost saving**, faster in ≥2/3 seeds, and endpoint noninferiority (lower CI >−1 pp). Apply the symmetric rule for slower; otherwise call efficiency inconclusive. Report censoring explicitly.
   - **Corroboration:** normalized trapezoidal AUC over the common **0.5M–6M** curve window, using actual logged decisions. Do not invent a zero-step score: the trainer does not log one. AUC is secondary, not a substitute if crossing results disappoint.
   - **Costs:** report PPO-only and total-cost crossings. Warm-up costs **90,112 decisions = 81,920 fitting + 8,192 diagnostic**; demonstrations cost **88,744**. Thus BC’s total-cost axis adds **178,856 decisions**. Count the full demonstration acquisition for a standalone run; disclose reuse across seeds separately. Report BC optimization time and evaluation cost separately. Any benefit is conditional on the single shared BC checkpoint.

6. **Shipping rule: agree with your proposal.** Among the five scratch-v2 seeds’ **eligible selected checkpoints**, ship the highest 100-selection-day medium/high mean. Break exact ties by fewer training decisions, then lower seed number. Never use screen results to break ties. Report all five seeds’ scores, mean/SD/range and eligibility failures, with the shipped winner identified separately; comparisons in items 4–5 remain matched three-seed comparisons. If none qualifies, report failure rather than relaxing the gates.

Read-only review; no files changed or control-result files inspected.
