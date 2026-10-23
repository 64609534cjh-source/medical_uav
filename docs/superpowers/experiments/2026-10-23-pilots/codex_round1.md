**I recommend freezing a deadline-focused scheduling benchmark before M2 data collection, using planner imitation → PPO as the main agent, and keeping scratch PPO as a required scientific control.** The project should explain whether learning can capture—and improve on—the planner’s decisions, with substantially cheaper inference.

I would overturn four parts of today’s draft: postponing reward changes, postponing the planner, excluding visualization from M2, and treating imitation as merely a fallback.

I read the supplied documents and code. Read-only verification passed **128 tests**, and reproduced Slack-aware’s development on-time rates: **99.26% / 80.99% / 65.89%**. No files were edited. Recommendations and experimental targets below are proposed, not completed results.

**1. Environment decisions: freeze by October 25, before collecting training demonstrations**

| Decision | Recommendation and reason |
|---|---|
| **Reward** | **+1 per on-time delivery; zero otherwise.** No energy penalty, expiry penalty, pickup bonus, or waiting penalty in the main experiment. Battery constraints already enforce safety. Energy and late completion remain reported outcomes. |
| **RETURN** | **Keep it.** It permits anticipatory repositioning and disposal at the center without compulsory charging. Replacing Slack-aware’s WAIT with RETURN tests one particular rule; its negligible effect does not establish that RETURN has no value to a learned policy. |
| **Charging** | **Keep non-interruptible target charging at 60/80/100%.** It is a clear commitment decision, already implemented and tested. Explicitly disclose that arrivals cannot interrupt charging. Interruptible charging belongs in a separately versioned sensitivity experiment, not a late change to the training environment. |
| **Map, speed, handling, demand** | **Keep existing values and map 0 for the primary benchmark.** Queueing, batching, battery depletion, and charging already produce substantial headroom. Artificially lengthening flights would change the question and invalidate calibration. |
| **Slots** | **Keep 12 pending + 3 cargo slots.** Existing evidence does not justify increasing them. Preserve aggregate backlog information and log hidden feasible orders under learned policies, not only Slack-aware. |
| **Observation** | **Add the complete 7×7 normalized flight-time matrix now.** The existing observation gives distances from the current location and some order-specific summaries, not all inter-node relationships. The extra 49 features make map-transfer experiments interpretable later. Total: **452 features**, with an 80-feature global block. |
| **Other semantics** | Keep expiry at twice the deadline, expired-cargo handling, event ordering, safety masks, always-valid WAIT, and draining outstanding orders after arrivals stop. Add no artificial action-count termination to normal training. |

The reward choice is deliberate. The current reward values a slightly late delivery almost like an on-time delivery; the headroom results show that this materially changes dispatch priorities.

The compromise is also a genuinely different objective. With every order resolved,

\[
N_{\rm on}+0.1N_{\rm late}-0.1N_{\rm expired}
=1.1N_{\rm on}+0.2N_{\rm late}-0.1N_{\rm released}.
\]

It rewards rescuing late orders sufficiently often to trade away an on-time delivery. That may be defensible operationally, but it should not silently govern a project whose headline objective is timely service.

**Run the three-reward planner screen anyway:** 40 existing development days × medium/high × three scores. Report on-time, completed, expired, energy, and per-type outcomes. Its purpose is to document the trade-off, not repeatedly tune coefficients until RL looks good.

Two qualifications belong in the formulation:

- On-time-only reward optimizes expected **count**. Pooled on-time fraction has a policy-independent released-order denominator; mean daily fraction weights days differently. Report both and retain mean daily fraction for continuity with M1.
- This is a synthetic, equal-order-priority benchmark. It does not establish clinically appropriate triage. A per-type collapse must be visible, not hidden by the aggregate.

Freeze a configuration identifier, observation/action schema, and reward definition. Preserve M1 under its submission tag; do not require new rewards to reproduce old numerical returns.

**2. Research question and final story**

> **Can a learned dispatcher improve timely medical delivery by coordinating batching and charging under uncertain arrivals, and how much of its performance comes from imitation versus reinforcement learning?**

The supporting question is whether a learned policy can approximate a strong online planner at lower decision cost.

The final story should have three connected findings:

1. **Objective design matters:** rewarding late rescues can improve return while hurting timely service.
2. **Planning establishes achievable headroom:** Slack-aware is useful but leaves measurable improvements available.
3. **Learning converts experience into a dispatch policy:** BC captures planning behavior; PPO’s incremental contribution is measured separately.

Pre-register these practical targets:

- **Useful improvement:** at least **+3 percentage points** over Slack-aware at medium load, paired 95% CI above zero; no more than 2 points worse at high load or 1 point worse at low load.
- **Strong learning result:** PPO improves its own BC initialization, consistently across training seeds.
- **Strong planning comparison:** recover at least half the planner–Slack-aware gap, or finish within 2 points of the planner while demonstrating a substantial measured inference-speed advantage.
- **Required explanation:** show which decisions account for differences—batching, charging timing, repositioning, or abandoning hopeless work.

A BC policy that matches the planner but gains nothing from PPO supports a **distillation result**. A scratch PPO failure is also reportable. Neither should be renamed an RL success.

The 30 ms prototype planning time is already small relative to minute-scale actions. Therefore faster inference is a computational advantage, not evidence that planning is operationally unusable.

**3. Roadmap, budgets, and submissions**

| Dates | Concrete work and acceptance | Submission versus internal work |
|---|---|---|
| **Oct 23–25** | Freeze environment; validate visible-information planner; reward screen; seed manifest; unified evaluator; charging/batching probes; readable replay prototype. | Internal evidence supporting M2 reflections. No training dataset before freeze. |
| **Oct 25–28** | Implement both networks; collect demonstrations; train BC; integrate masked PPO and complete a short learning run; validate checkpoint loading on CPU. | Build M2 around the actual final agent interface. |
| **Oct 29–30: M2** | Extracted-zip installation and demo rehearsal; finish the two-page report. Internally, PPO must already run end-to-end. | Submit commented environment/network/BC code, trained BC checkpoint, forward/backward demo, lightweight evaluation command, exact environment, README, readable replay, report. Exclude unfinished PPO, scratch experiments, large datasets, and design drafts. |
| **Oct 31–Nov 6: M3** | Complete three seeds each of scratch PPO and planner-BC→PPO; compare against frozen BC and conventional methods; learning curves, probes, failure analysis. | Submit learning/training/evaluation code and checkpoints according to the actual M3 rubric once released. Do not invent its report length or packaging requirements. |
| **Nov 7–11** | Extend main methods to five seeds; complete architecture/teacher controls, stronger planner comparison, and transfer study. Freeze configurations and checkpoint-selection rules. | Internal development results; final reproducibility commands already working. |
| **Nov 12–14** | Locked test evaluation, uncertainty estimates, behavioral case studies. | Produce final tables and figures. End performance tuning before viewing test results. |
| **Nov 15–17** | Final video; individual report drafts; clean installation on another machine; student walkthroughs. | Assemble complete submission candidate. |
| **Nov 18–20** | Correctness/package fixes, final checks, submission buffer. | One reproducible final package, trained models, video, and individual ≤10-page reports. |

**Training budget**

Use training seeds **11, 22, 33, 44, 55**.

| Experiment | Budget |
|---|---:|
| Integration and targeted PPO pilots | Up to four × 1M decisions |
| Set-network scratch PPO | Five × 5M |
| Set-network planner-BC→PPO | Five × 5M |
| MLP planner-BC→PPO architecture control | Three × 5M |
| Set-network Slack-aware-BC→PPO teacher control | Three × 5M |
| Secondary mixed-map training | Three × 5M |

That is **99M planned environment decisions**, plus demonstration generation. It is a budget, not a requirement to spend samples after a correctness failure. Measure actual end-to-end throughput before translating it into hours.

Allow one predeclared extension from 5M to 10M for **both main methods** if development curves still improve by more than 1 point over the final million decisions. Do not selectively extend a disappointing seed.

**Data and statistics**

- Keep demand seeds below 100,000 for training. Reserve distinct blocks for demonstrations and online RL.
- Existing **100000–100039**: environment screening and debugging.
- Fresh **100040–100139**: model/checkpoint selection.
- Final **200000–200299**: 300 paired days per load; expand the current manifest before training.
- Same map, order stream, constraints, and observation access for all methods.
- Select one checkpoint per training seed for all loads: maximize mean medium/high development on-time rate, subject to low-load ≥97% and no per-type development rate more than 3 points below Slack-aware. If no checkpoint qualifies, report the failed gate; do not quietly relax it.
- Bootstrap **whole days**, paired across methods, with 10,000 resamples. Report each training seed and its spread. Five models on 300 days are not 1,500 independent test days.
- For uncertainty across both days and trained models, also resample the crossed day and training-seed dimensions.
- Report per-load results, per-type counts/denominators, completion, expiry, energy, charging, batching, and decision latency.

Architecture and teacher controls use three seeds and support narrower claims than the five-seed primary comparison. Keep exploratory findings explicitly labeled.

**4. Network, agent, teachers, and conventional baselines**

**Network.** Keep the proposed small set/attention architecture as the main candidate, with the enlarged global input:

- Pending encoder: `27→64→64`; cargo: `16→64`; global: `80→64`.
- One four-head, width-64 cross-attention layer; global query; pending/cargo tokens and an always-valid null token.
- Shared PICK scorer; separate heads for the other 14 actions and value.
- No positional encoding, dropout, recurrence, or transformer stack initially.
- MLP control: separate actor/critic, `452→256→256→output`.

Test pending permutation **equivariance**, cargo permutation invariance, empty states, padding, and masked distributions. Permuting slots must permute the associated PICK mask and targets too.

Keep infeasible-but-present orders in attention: they may become feasible after unloading or charging. Mask actions at the distribution.

Use the set network by default. Switch the production agent to MLP only if matched three-seed pilots show it ahead by more than 2 points on medium/high development performance with a paired CI excluding zero, or reveal unresolved set-network instability. Record that selection; do not generalize architectural superiority from one seed.

**BC.** Use the **validated visible-order rollout planner** as the main teacher from M2. Slack-aware is the cheap sanity-check teacher and an explicit control.

- Collect **1,200 planner episodes**, balanced across loads; roughly 100k–150k decisions.
- Split 80/20 by complete episodes, grouping any reused demand seed across loads.
- Adam `3e-4`, batch 256, up to 30 epochs; masked cross-entropy.
- Store observation, mask, chosen action, episode identifier, and candidate rollout scores.
- Evaluate closed-loop performance, action-family accuracy, and planner-score regret. Accuracy alone penalizes harmless ties and misses compounding errors.
- If BC trails its teacher by more than 3 points at medium/high, perform **one 300-episode learner-state relabeling round** using training seeds. Preserve a BC-only checkpoint after that round so PPO is compared with the actual initialization it received.

The planner should choose deterministically among equal primary scores. Store those ties; optionally use uniform targets over equally preferred actions rather than inventing a uniquely correct label.

**PPO.** Implement the standard clipped objective in readable PyTorch, following the [original PPO formulation](https://arxiv.org/abs/1707.06347), with consistent [invalid-action masking](https://arxiv.org/abs/2006.14171):

- `γ=1`: finite-day, undiscounted service objective.
- GAE `λ=0.95` per decision. Explain that this controls estimation, not discounting per minute.
- 16 environments × 128 decisions; four update epochs; minibatch 512.
- Clip 0.2; Adam `3e-4`; gradient norm clip 0.5; value coefficient 0.5.
- Fixed reward scaling `/10`; normalized advantages; fixed observation scaling.
- Entropy coefficient `0.01→0.001`; log actual entropy and valid-action counts.
- Stop remaining update epochs if approximate KL exceeds 0.02.
- Store collection-time masks/log probabilities; never rebuild masks from later states.
- Bootstrap at rollout boundaries; zero bootstrap at true termination; never carry GAE across resets.
- Assert invalid-action count is zero—the environment’s WAIT fallback otherwise conceals errors.

Before BC→PPO, collect a small **on-policy** batch from the cloned policy and warm-start the value head while freezing policy-affecting parameters. BC’s untrained value head must not be mistaken for a learned critic.

No permanent imitation loss in the primary PPO experiment: that would introduce another algorithmic question.

**Curriculum.** Scratch PPO gets at most 250k low-load decisions, then the fixed balanced mixture of all three loads. BC→PPO starts on that mixture immediately. Avoid long low→medium→high stages that repeatedly change the problem. One policy handles all loads.

**Baselines.**

| Method | Role |
|---|---|
| Random, EDF | Sanity checks and M1 continuity |
| FCFS | Oldest eligible pickup/delivery; shared charging fallback |
| Nearest | Minimum immediate travel + actual handling/disposal time; deterministic ties |
| Slack-aware | Strong interpretable rule baseline |
| Visible-order rollout | Required strong conventional planner and BC teacher |
| Scenario rollout | Stronger anticipatory conventional reference |

For the visible planner, evaluate every valid first action, then complete currently visible orders with Slack-aware. Exclude hidden-order details and actual future arrivals. Score future outcomes from the copied state, excluding irrelevant historical reward. Resolve ties lexicographically—on-time count, completion count, lower energy, then base-policy preference.

For scenario rollout, sample future arrivals from the **known demand model using a separate RNG**, sharing scenarios across candidate actions. Compare 4 and 16 scenarios over a fixed next-60-minute arrival window, then drain those simulated orders. The continuation policy only observes arrivals when revealed. Select the strongest development configuration meeting a measured p95 decision budget of one second.

**Do not call the existing one-step prototype “state of the art.”** Rollout has established routing foundations, and anticipatory methods are supported by work such as [Ulmer et al.’s offline–online dynamic routing method](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2017.0767). Our implementation should be described as a literature-grounded adaptation, with its differences stated. That gives a credible conventional comparison; the bonus cannot be guaranteed by labeling it SOTA.

**Generalization.** Primary claims concern new demand days on map 0. For the secondary study:

- Train a separate mixed-map version on maps 1–20, with its own demonstrations.
- Develop on maps 100–104.
- Test on maps 300000–300019 × ten demand seeds × medium/high.
- Compare map-0-only and mixed-map policies against the same baselines.
- Bootstrap maps as clusters.

The added matrix makes geometry available; it does not make a map-0-trained network automatically generalize.

**5. Visualization: fix the actual comprehension problem in M2**

The current demo renders one simulated minute per frame at 12 fps: **12 simulated minutes disappear every second**. A two-minute loading action lasts about 0.17 seconds. Merely increasing image quality will not fix that.

Deliver two views from the same event logs:

- **Explanation replay:** 45–90 seconds covering one short decision sequence. Show the order, deadline, selected action, battery/cargo consequences, and outcome. Hold important decision frames for 2–4 seconds.
- **Day overview:** accelerated timeline with battery, pending queue, charging intervals, and cumulative outcomes; clearly label playback speed.

Keep video fps independent of simulated-time speed. Add pause/scrub controls to a lightweight local replay if practical; MP4 remains the portable submission artifact.

For the final video, target 3–4 minutes:

1. Explain the task and constraints.
2. Show Slack-aware and learned policy on the **same arrivals, synchronized by simulated time**.
3. Explain one batching decision and one charging decision.
4. Show aggregate results and one failure/limitation.

Use persistent order IDs, large labels, and explicit late/expired status. Display the policy’s visible queue separately from any analyst view of the full backlog. Neural attention weights are not sufficient evidence of why a decision occurred.

Choose the main example by a documented representative-day rule, and label any deliberately selected success/failure case. A viewer should understand one consequential decision after one viewing.

**6. Report bookkeeping**

Maintain a small experiment ledger and decision log, linked to commits:

| Record | Minimum contents |
|---|---|
| Decision | Previous design, new choice, reason, evidence, affected milestone claims |
| Experiment | Hypothesis, config/schema hash, code commit, seeds, budget, outcome, conclusion |
| Dataset | Teacher/version, allowed information, generation seeds, split, relabeling history |
| Model | Initialization, training seed, optimizer state, checkpoint-selection reason, checksum |
| Results | Per-day raw counts and metrics, aggregation script, uncertainty method |
| Provenance | Libraries, borrowed algorithm ideas, AI contributions, human checks |
| Reflection | What failed, what assumption was wrong, what changed because of evidence |

Explicitly preserve these changes from earlier submissions: reward, map features, planner teacher, BC→PPO as primary, fixed-map primary scope, and any later architecture selection.

For M2, use page one for network/dataflow/masking/demo and page two for learning evidence, libraries/AI disclosure, and reflections. Demonstrate a separate value loss/backward pass if the report claims to exercise the critic; BC cross-entropy alone does not train it.

Start individual final-report outlines during M3. Each member should explain the formulation, reproduce a result, and interpret a failure. Record real contributions; do not assign fictional coding roles.

Keep the final evaluator in the submitted package—the current plan to export-ignore evaluation should not survive into the final release.

**7. Cheap checks that could change the plan**

| Check | Budget | Result and consequence |
|---|---:|---|
| **Planner information audit** | 100 saved development states | Change actual future arrivals and hidden-order details while preserving public information. Any change in planner choice blocks demonstrations until fixed. |
| **Planner-score audit** | Same states | The current `ontime` implementation adds `0.001 × old reward`; `late<w>` retains a −0.5 expiry cost. Neither directly implements the proposed pure/compromise comparison. Correct scoring before interpreting new results. |
| **Reward comparison** | 240 planner-days | Quantify timely-service versus late-rescue trade-offs. A severe per-type failure changes the claimed applicability and requires an explicitly separate priority-sensitive objective, not covert coefficient tuning. |
| **Charging commitment diagnostic** | 40 paired days/load | Log arrivals during charging and inspect avoidable misses; optionally compare an interruptible variant before freeze. A >2-point improvement with CI above zero would justify revisiting charging before data collection; otherwise retain the macro design. |
| **Slot sensitivity** | 40 medium/high paired days | Recheck visible-12 versus larger visibility with the finalized planner. Increase slots before collection only for >1-point improvement with CI above zero; otherwise retain 12. |
| **Action diversity and teacher coverage** | Demonstration pilot of 100 episodes | Missing charge targets/RETURN or poor urgent-order coverage triggers targeted training-state collection, not fabricated performance claims. |
| **PPO numerical checks** | Toy trajectories + 50k decisions | Ratio≈1 before updates, correct GAE boundaries, finite gradients, no invalid actions. Any failure stops long runs. |
| **Actual throughput** | 100k decisions/configuration | Compare synchronous CPU, limited async workers, and GPU batches. Select measured throughput; the CPU-only Torch installation currently cannot use the A5000. |
| **Fresh installation** | One clean Linux install; teammate Windows demo | Any missing wheel/import/path changes packaging immediately. Pin tested versions and distinguish CPU reproduction from GPU training requirements. |

The already-completed read-only checks support the current simulator, not the new reward/network/planner implementation.

Charging deserves attention: my reproduction measured median charging-action duration, including travel, of **12.20 minutes at medium** and **12.35 at high**. There were **512/614 arrivals during those actions across 40 days**. These are exposure counts, not proof that interruption improves service.

**8. Top risks and cheapest guards**

| Risk | Cheapest effective guard |
|---|---|
| PPO adds nothing beyond imitation | Preserve BC-only checkpoints and compare paired performance before/after PPO. |
| Reward improves while the claimed objective worsens | Main reward directly counts on-time deliveries; always show completion and per-type outcomes. |
| Planner benefits from privileged information | Restricted public-state construction plus future/hidden-state invariance tests. |
| Agent stalls or abuses WAIT/charging | Small urgent-order and low-battery probes; log elapsed idle time, repeated actions, and missed opportunities. |
| BC looks accurate but fails when controlling the UAV | Closed-loop evaluation and one bounded learner-state relabeling round. |
| Development overfitting | Separate old calibration days, fresh model-selection days, and locked final tests. |
| Architecture or teacher claim is confounded | Matched budgets, same data where applicable, multiple seeds, and explicit compute accounting. |
| Map-transfer claims exceed evidence | Complete map input, separate mixed-map training, unseen-map evaluation, map-cluster uncertainty. |
| Package works only on the development machine | Extracted-archive CPU rehearsal before every submission. |
| Technically correct work remains incomprehensible | Fix replay now; require student walkthroughs and one-view comprehension of the final examples. |

**The priority through November 20 is a reproducible, explainable comparison of rules, planning, imitation, and RL.** More architecture or simulator complexity earns its place only when it resolves a specific unanswered question.