# Final phase: pre-registered energy study, scenario rollout and locked test (2026-10-27)

Designed by Codex (astra high) from roadmap §1–§10 and the M3 results; adopted as written and committed before any of these experiments ran. Scope changes: planner-BC → PPO and longer PPO continuation are dropped (item 4). Schedule: item 5 (29–30 Oct reserved for the M2 report and submission). Amended 10/27, after the registration and before any multi-map result: mixed-map training was added at the owner's request with its own registration (`2026-10-27-multimap-preregistration.md`); both the single-map and the multi-map agent enter the locked test.

Keep the shipped PPO and existing controls frozen. The final technical contribution should be **one controlled energy study, one audited scenario planner, and one locked evaluation**. Do not start another training campaign.

I read the requested [roadmap](/home/caijihang/projects/medical_uav/docs/superpowers/specs/2026-10-23-roadmap-to-final.md), [M3 review](/home/caijihang/projects/medical_uav/docs/superpowers/experiments/2026-10-24-m3/codex_final_review_m3.md), planner, observation, demand model, and simulator. Branch is `m3-ppo`; no files changed or evaluation episodes run. The specifications below are proposed preregistration text: commit them before collecting the new results.

1. **Energy intervention: test anticipatory charging with an always-full control; use PPO masking as corroboration.**

   A full-charge mask alone cannot establish that *early* charging explains the gain: it removes an action throughout the day and changes the trained policy’s state distribution. Use the following four conditions.

   | Condition | Exact definition | Question |
   |---|---|---|
   | **Slack** | Existing policy, unchanged | Reference |
   | **Slack-100** | Preserve Slack’s priorities, charging triggers and charger ordering; change charge-target preference from `(80,100)` to `(100,80)` | Does a higher target help without adding anticipation? |
   | **Slack-100-prepeak** | Slack-100 plus the fixed override below | Does adding anticipatory charging help beyond full-charge preference? |
   | **PPO-no100** | Frozen PPO, greedy inference, with both `CHARGE(*,100)` actions removed from its inference mask | Does the trained controller depend on access to full charging? |

   Apply PPO-no100 to **all five selected scratch checkpoints**, pairing each with its unchanged counterpart. Identify seed 11 separately as the shipped controller. Do not retrain.

   **The pre-peak override is fixed at \(B=80\%\), \(W=30\) minutes.** For each peak start \(p\in\{60,300\}\), allow at most one override when:

   - \(p-30\le t<p\);
   - current battery is strictly below 80%;
   - the UAV carries **no boxes**, including expired cargo;
   - a valid full-charge action can finish by \(p\).

   For each charger \(c\), calculate, with empty-cargo energy consumption,

   \[
   t_{\mathrm{finish}}(c)=t+\tau(x,c)
   +\frac{100-[b-E(x,c,0)]}{5}.
   \]

   Among valid actions satisfying \(t_{\mathrm{finish}}\le p+\mathrm{EPS}\), choose the earliest completion; ties go to the lower node index. Execute it before Slack-100’s ordinary choice, and mark that peak’s override as used. Otherwise use Slack-100. Reset both flags each episode.

   This intervention can postpone pickups, but cannot strand cargo during a discretionary top-up. The parameters have direct operational interpretations; **do not search over them**.

   **Data discipline.** Run the fixed conditions on selection days `100040–100139`, then screen days `100000–100039`. Report both. Screen is a development replication, not an untouched confirmatory sample: its behaviour already motivated this study. Final confirmation uses the locked test. If the intervention triggers rarely or performs badly, retain that result; do not revise \(B,W\) after seeing it.

   **Primary estimand.** For each day and load, calculate the on-time fraction among **all orders released inside either peak**, regardless of when they are delivered. Define

   \[
   \Delta_{\mathrm{timing}}
   =\tfrac12\sum_{\ell\in\{\mathrm{medium,high}\}}
   \operatorname{mean}_d
   \left[
   P_{\mathrm{Slack100prepeak},\ell,d}
   -P_{\mathrm{Slack100},\ell,d}
   \right].
   \]

   Bootstrap whole day IDs jointly across medium/high, 10,000 resamples. Report each load separately alongside this registered combined endpoint.

   Register these interpretation rules:

   | Result | Permitted conclusion |
   |---|---|
   | Timing gain ≥1 percentage point, 95% CI above zero, peak-start battery rises ≥5 battery points on average, and absolute charging minutes inside peaks decrease | Anticipatory top-ups improve peak service in this rule family, with the predicted energy changes |
   | Manipulation changes energy as intended, but the gain’s CI upper bound is below +1 point | This intervention does not reproduce a practically meaningful peak-service benefit |
   | CI upper bound below zero | The intervention harms peak service |
   | Few overrides or little change in peak-start battery | Failed manipulation; weak evidence about the underlying mechanism |
   | Anything else | Inconclusive |

   Register **secondary contrasts**: Slack-100 minus Slack; prepeak minus original Slack; original PPO minus PPO-no100. For PPO masking, report the paired effect for each training seed and the average across seeds.

   Also report all-day on-time, off-peak on-time, per-type service, override frequency, battery at each peak start, absolute peak charging minutes, total charging minutes, distance, and full-charge frequency. Require the lower CI for the prepeak policy’s all-day medium/high difference versus Slack to exceed −1 point before calling it an overall useful improvement.

   **Interpretation boundary:** these experiments estimate effects of changing charging policies in this simulator. Even joint positive results do not identify what fraction of PPO’s advantage is mediated by battery, or prove that its learned timing is optimal.

   **Write these tests first:**

   - Mask exactly actions 20 and 23; preserve the original environment mask and observation; WAIT remains valid.
   - Slack-100 changes target preference only, preserving existing unloading/batching priorities and charging triggers.
   - Cover \(t=p-30\), \(t=p\), \(b=80\), occupied/expired cargo, unreachable chargers, and a charge finishing just after the peak starts.
   - Enforce one override per peak and reset flags between episodes.
   - Peak cohorts use release time with half-open windows `[60,180)` and `[300,390)`; retain numerator and denominator counts.
   - Compute charging overlap by segment intersection and battery at peak boundaries using the existing interpolation convention.

2. **Scenario rollout: implement arrival-sampling rollout over Slack-aware, with an explicit hidden-backlog limitation.**

   The existing [rollout planner](/home/caijihang/projects/medical_uav/medical_uav/baselines/rollout.py) evaluates every valid first action and lets Slack-aware resolve visible work. Extend that idea with sampled future arrivals.

   **Information contract.** Give the new planner an observation-and-mask interface, plus immutable public model constants and its own random stream. It may use:

   - Current time, location, battery, cargo, and the 12 visible orders.
   - The observed travel-time matrix and load-rate feature.
   - Public physics, order-type distributions, reward coefficients, and fixed peak schedule.
   - Aggregate backlog and recent-arrival features already present in the observation.

   It must not read the real demand seed, environment RNG, future-order list, hidden-order records, or historical order/event records unavailable in the observation.

   **Sample future arrivals only. Do not impute existing hidden pending orders.** Aggregate counts do not determine their ages, destinations, or conditional distribution after earlier dispatch decisions. A credible hidden-backlog belief model would be a separate project.

   For this version, retain backlog aggregates for diagnostics but do not turn them into synthetic jobs or adjust the known arrival rate. Report how often hidden backlog exists, and performance conditional on overflow as a descriptive limitation. This planner uses a subset of PPO’s information, plus a known transition/demand model; do not describe it as a complete belief-state planner.

   **Exact scenario generation.** At a decision at time \(t\), set \(T=t+60\). For each scenario, sample arrivals in

   \[
   (t,\min(T,480)).
   \]

   Split this interval at peak boundaries. On each segment \([a,b)\), draw

   \[
   N\sim\mathrm{Poisson}(\lambda\,m(a)(b-a)),
   \]

   then draw \(N\) uniform release times. Sort releases. Draw type, route and weight using exactly the distributions in [orders.py](/home/caijihang/projects/medical_uav/medical_uav/env/orders.py): type shares `.20/.30/.30/.20`, the specified route categories, uniform hospital choices, and type-specific uniform weights. Deadlines and expiry follow the type constants.

   Use independent per-scenario RNG streams derived from a fixed planner seed, episode counter, decision counter and scenario index. **Never derive them from `env.demand_seed`.** Freeze master seed `271828`. Make scenarios 1–4 identical between the 4- and 16-scenario versions.

   Generate each scenario **once before evaluating actions**. Every candidate receives the same releases and marks for that scenario.

   **Action evaluation.**

   1. Construct a fresh planning state from observable data: visible orders and cargo with local synthetic IDs, current physical state, and the sampled future stream. Exclude historical rewards and resolved orders.
   2. Enumerate every action allowed by the real root mask. Preserve the root PICK-slot mapping.
   3. Execute candidate \(a\) completely.
   4. Follow the **original Slack-aware policy**, unchanged, at subsequent decision epochs. It sees only the planning simulator’s current visible slots and cargo; sampled future orders become visible only upon release.
   5. Accumulate the unscaled v2 reward: on-time delivery \(+1\), late delivery \(+0.1\), expiry \(-0.1\), flight energy \(-0.001\) per battery point.

   Use a **60-minute arrival horizon with a Slack terminal rollout**: after \(T\), introduce no additional arrivals, but let Slack finish or expire the remaining modelled work. Stop when no sampled future arrivals, pending orders, or live onboard orders remain. Do not idle until the real day’s termination merely because `Simulator.is_done()` still expects the release window to end.

   This specifies terminal value without inventing a battery bonus or treating unresolved orders as worthless:

   \[
   \hat Q_S(a)=\frac1S\sum_{s=1}^{S}
   \left[
   R_{\text{through horizon}}(a,\omega_s)
   +V_{\text{Slack, no further arrivals}}(x_{T,s})
   \right].
   \]

   Actions crossing \(T\) complete normally; account for their rewards once. Describe this honestly as **60-minute stochastic lookahead with a terminal rollout**, since simulated resolution can extend beyond 60 minutes.

   Choose the largest sample-average value. Within `1e-9`, prefer original Slack’s root action; otherwise choose the lowest action index. Execute only that first action in the real environment and replan.

   **Method attribution.** Call it *Monte Carlo rollout over Slack-aware, using sample-average action values and common random scenarios*: an adaptation of [Bertsekas’s rollout policy-improvement framework](https://web.mit.edu/dimitrib/www/ADP-MPC.pdf). Bent and Van Hentenryck’s [multiple-scenario approach](https://pubsonline.informs.org/doi/10.1287/opre.1040.0124) is related motivation, but their method maintains routing plans and uses a consensus decision mechanism. This implementation does not reproduce that algorithm. Claim neither state-of-the-art status nor a guaranteed improvement under finite sampling and an approximate state model.

   **Runtime and selection.**

   - Start with all valid actions and \(S\in\{4,16\}\); no candidate pruning.
   - Build one sanitized root snapshot and one template per scenario. Clone only mutable simulation state for each candidate.
   - Reuse immutable map/model data. Avoid deep-copying the real environment’s full history.
   - Provide a planning step that updates simulator state, slots and masks without rendering, metrics, decision logs or the 452-feature observation at every inner step.
   - Verify this fast path against the ordinary simulator on identical sanitized states before relying on it.
   - Use one CPU thread per planner worker. Parallelize independent episodes, with **at most eight workers** on the shared machine. Do not create a process pool inside each decision.

   Benchmark each candidate serially on at least 1,000 decisions per load, including decisions encountered by that candidate itself. Run three passes; include scenario generation, cloning and terminal rollouts in timing. **Eligibility requires decision-level p95 ≤1 second on every load in every pass.** Aim below 0.8 seconds for contention margin.

   Among eligible versions, select on all 100 selection days:

   - Use the roadmap’s low-load and per-type eligibility gates.
   - Choose 16 only if its medium/high mean daily on-time score exceeds 4 by at least **0.5 percentage points**.
   - Otherwise choose 4. If only one qualifies, choose it.
   - If neither satisfies service gates, report that failure and retain the better latency-feasible version as a comparator, without claiming it meets the service target.
   - If neither satisfies latency, optimize implementation and remeasure before freezing; do not hide the failure with partial action evaluation.

   Freeze the selected version, then run screen confirmation. Show the 4-versus-16 development comparison regardless of which wins. No parameter adjustment after screen results.

   **Information audit and tests required before selection:**

   | Test | Required result |
   |---|---|
   | 100 distinct saved decision states, including overflow states | Assert state uniqueness and actual hidden-order coverage |
   | Same observation, mask and planner RNG state; mutate actual future orders, hidden details, demand seed and environment RNG | Identical generated scenarios, action scores and chosen action |
   | Hidden-order mutations | Preserve observable counts and same-destination features when claiming identical observations |
   | Poison forbidden environment attributes | Planner still operates; preferably it receives no environment object |
   | Candidate enumeration reordered | Identical scenarios, scores and winner |
   | 4 versus 16 | Exact common prefix of four scenarios |
   | Input state/RNG immutability | Real environment and its RNG are unchanged |
   | Expiry/release ties, charging across \(T\), expired cargo, release-window end | Match simulator semantics; no double counting or artificial early termination |
   | No sampled arrivals | Recover visible-work rollout values, within declared reconstruction tolerance |
   | Selected action versus Slack on the same samples | Selected sample-average score is no smaller; no claim about true expected reward |

   The observation is float32. Treat the supplied root mask as authoritative, explicitly test reconstruction near feasibility boundaries, and never let the environment’s “invalid action becomes WAIT” fallback conceal a planner error.

3. **Locked test: freeze a complete method manifest, then evaluate once.**

   Keep map 0, environment v2, and all three loads. Use every day `200000–200299` for every method. Evaluate all controllers under **v2 reward**, including policies trained with alternative rewards.

   **Frozen entrants:**

   | Group | Methods/checkpoints |
   |---|---|
   | Conventional references | Random, EDF, FCFS, nearest-first, original Slack-aware |
   | Planning | Existing audited visible-order rollout; selected scenario rollout |
   | Primary learned agent | All five already-selected scratch set-PPO checkpoints; seed 11 identified as shipped |
   | Existing imitation evidence | Shipped Slack-BC set and MLP checkpoints |
   | Existing matched controls | Selected MLP PPO, M1-reward set PPO, on-time-only set PPO, Slack-BC→PPO: seeds 11/22/33 each |
   | Energy interventions | Slack-100, Slack-100-prepeak, PPO-no100 for each of the five primary checkpoints |

   That is **33 configurations**, with seed 11 counted once, or 29,700 day episodes across three loads. Put the main comparisons in the report body and control details in compact tables or accompanying evidence.

   **Statistics.**

   - Primary operational metric: **mean daily on-time rate**, separately by load.
   - Report pooled on-time/served/expired counts and pooled per-type rates alongside it.
   - Use paired whole-day bootstrap, **10,000 resamples, fixed bootstrap seed 0**, for PPO versus Slack and each planner, and the registered intervention contrasts.
   - For the five-seed algorithm result, average across training seeds **within each day**, then bootstrap days. These intervals describe day uncertainty conditional on the five trained policies.
   - Separately report all five seed means, sample SD and range. Do not treat 1,500 seed-day rows as independent days.
   - For three-seed controls, retain matched seed comparisons and report each seed difference.
   - Recompute pooled ratios from resampled numerator/denominator counts. Do not bootstrap individual orders as independent observations.
   - For the energy study’s combined endpoint, resample day IDs jointly across medium/high. Report any zero-denominator exclusions explicitly.

   Report the shipped controller’s results separately from the five-seed mean; do not present its selected best-seed performance as typical training performance.

   Preserve the roadmap targets: medium gain over Slack ≥3 points with CI above zero; report high/low noninferiority against the −2/−1-point margins. For each planner, assess the −2-point comparison margin separately at medium and high. Identify intervals as per-comparison 95% intervals; treat unregistered subgroup findings as exploratory.

   **Report operational evidence:** return, energy, distance, total and peak charging minutes, peak-start battery, boxes per stop, WAIT/RETURN shares, per-type denominators, peak/off-peak service, and decision latency. Retain individual timing samples: the current evaluator’s per-day p95 values cannot reconstruct an overall decision-level p95. Keep throughput-run timings separate from serial latency benchmarks.

   **Freeze conditions—all must pass before opening test days:**

   - Commit/hash every policy, selected checkpoint, intervention rule, planner configuration and analysis script.
   - Record environment/configuration, dependency versions, selection outcomes, RNG rules and thread settings.
   - Pass information audits, action-validity checks, metric tests, and planner latency acceptance.
   - Complete the planned development analyses and record failures as well as successes.
   - Validate bootstrap/report generation using development data.
   - Rehearse evaluation from the exported submission package.
   - Add explicit locked-test access: the current evaluation CLI exposes only screen/select, which is appropriate until this point.

   Run one manifest-driven evaluation campaign, saving results atomically by method/load/day so interrupted jobs can resume without reselection. Do not replace checkpoints or change methods after viewing test outcomes. A substantive post-test bug invalidates affected results; disclose it and the corrective rerun rather than claiming the original test remained untouched.

4. **Remaining work, ranked by value per CPU-hour.**

   CPU costs below are planning estimates, not measured promises. Count a one-hour training run on four threads as approximately four core-hours.

   | Rank | Work | Decision |
   |---|---|---|
   | 1 | Energy interventions and mechanism diagnostics | Do. Cheap inference; directly addresses the remaining causal question |
   | 2 | Failure cases and distance accounting | Do. Mostly analysis of existing/development traces |
   | 3 | Side-by-side video and exported-package rehearsal | Do. Required deliverables with little experimental compute |
   | 4 | Scenario rollout and locked comparison | Do. Largest remaining evaluation cost, but necessary conventional comparator |
   | 5 | Small unseen-map stress test | Optional, after core methods pass; no retraining |
   | 6 | Planner-BC→PPO | Drop from this final phase |
   | 7 | Longer PPO continuation | Drop |
   
   **Failure study:** choose cases by a rule, not visual appeal. On high-load screen days, select the day nearest the median shipped-PPO-minus-Slack daily difference, the worst difference, and a plasma-heavy failure case selected by the largest plasma on-time shortfall. Use lowest day ID for ties. Explain decisions and outcomes without changing policies.

   Decompose flight distance into measurable categories: empty versus loaded legs, destination/action categories, charger travel, and pickups yielding no successful delivery. Make categories mutually exclusive where summing them. Avoid calling every empty leg “wasted”; repositioning can be useful.

   **Video:** use shipped PPO versus Slack-aware on the same arrivals, synchronized by simulated time. Show battery, charging, visible orders and deadlines; pause at one charging decision and one service decision. Include aggregate locked results and one failure. Use a development-selected day so the demonstration is not chosen for an unusually favourable test outcome. Target the roadmap’s 3–4 minutes.

   **Optional map stress:** freeze map seeds 1–10, selection demand seeds `100040–100059`, medium/high loads, five PPO checkpoints, Slack and visible rollout. Use no adaptation or checkpoint reselection. Report map-level effects and uncertainty clustered by map; acknowledge ten maps give limited precision. If map generation fails, report it rather than selecting replacements based on performance.

   **Why drop planner imitation:** it would answer a new teacher-specific learning question, requiring demonstrations, BC, warm-up, three PPO runs and selection. Existing scratch PPO already exceeds that teacher, and Slack-BC→PPO has not established an efficiency benefit. Record this as a deliberate roadmap scope change, not a completed experiment.

   **Why drop continuation:** five-seed evidence is stronger than a late extension of the winning seed. A credible continuation comparison would require matched seeds, a registered optimizer/schedule policy and further selection. It would delay more valuable evidence.

   Also drop new architectures, hidden-backlog inference, DAgger, hyperparameter sweeps and mixed-map training.

5. **Ordered schedule.**

   | Date | Work and completion gate |
   |---|---|
   | **27 Oct** | Commit the energy and scenario preregistrations. Write intervention boundary tests, peak-count metrics and the planner information-contract tests. Implement the energy conditions and run selection-day evaluation. Begin the sanitized planner-state adapter |
   | **28 Oct** | Freeze energy implementation and run screen replication. Produce the energy table with manipulation checks. Implement reference 4-scenario rollout, exact demand sampling, common-scenario tests and terminal scoring. Measure an initial runtime profile; make no promise yet about 16 scenarios |
   | **29 Oct** | Reserved for M2 report and package rehearsal |
   | **30 Oct** | M2 submission and buffer |
   | **31 Oct** | Finish scenario reference implementation and 100-state audit. Implement the fast planning path only where profiling justifies it |
   | **1 Nov** | Verify fast/reference agreement; benchmark 4 and 16 scenarios, including full terminal-rollout cost |
   | **2 Nov** | Complete selection-day comparison and select the scenario count using the registered rule |
   | **3 Nov** | Screen confirmation; freeze planner behaviour. Extract failure cases and distance decomposition |
   | **4–5 Nov** | Write M3 report from completed M3 evidence; finish portable provenance/evidence and rehearse the actual M3 archive. New final-phase results are supplementary |
   | **6 Nov** | Submit M3; buffer |
   | **7 Nov** | Optional fixed map stress test only if core gates passed. Otherwise finish audits/packaging |
   | **8 Nov** | Freeze final method manifest, metrics, analysis and checkpoint hashes. Dry-run the full reporting pipeline on development data |
   | **9–11 Nov** | Execute the locked test campaign with up to eight single-thread workers. Run separate serial latency benchmarks |
   | **12–14 Nov** | Produce paired intervals, seed-spread tables, intervention conclusions and final figures. Draft individual reports within ten pages |
   | **15–17 Nov** | Produce the synchronized comparison video, complete reports, and rehearse installation/demo/evaluation from the exported final package |
   | **18–20 Nov** | Submission fixes and buffer; no new experiments or policy selection |

   Budget scenario evaluation from measured mean decision time:

   \[
   \text{core-hours}\approx
   \frac{\text{episodes}\times\text{decisions per episode}\times
   \text{mean decision seconds}}{3600}.
   \]

   For example, 900 episodes at 100 decisions and 0.25 seconds per decision cost about **6.25 core-hours** for one planner. Measure this on 28 October and update the execution budget—not the scientific protocol.
