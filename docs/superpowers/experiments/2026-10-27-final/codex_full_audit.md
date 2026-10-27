# Codex: whole-project audit before the locked test (2026-10-27)

Codex (gpt-6-astra, high), read-only; prompt: the full project, specs, results and code. Verbatim.

1. **The central result is credible, but the final scientific claim is not yet established.**

   This is a substantial course project: a coherent simulator, an independently implemented masked PPO, meaningful conventional comparators, replicated training, explicit checkpoint selection, and unusually candid reporting of negative findings. Its strongest defensible claim is currently: **selected PPO policies improve timely service over the implemented rules and rollout planners on map-0 development days.** “PPO beats all baselines” requires qualifications about load, evaluation population, selection, and planner strength.

   I read the requested specifications, README, all experiment reviews/preregistrations, and relevant implementation and results. I independently checked all **17 saved checkpoint selections**, the shipped checkpoint’s checksum, the main comparisons, and the energy analysis. **211 tests passed across two runs**; rendering/package-import checks encountered read-only cache restrictions, and file-writing tests were excluded. I changed no repository files and did not access locked test days.

2. **The formulation is defensible as a synthetic dispatch benchmark, but calling the 452-vector a fully observed MDP state would be incorrect.**

   Actions have variable durations and execute to completion. The natural formulation is a **partially observed, event-driven semi-Markov dispatch problem**. The complete operational state can support a Markov formulation; the supplied observation cannot, because hidden pending orders retain unobserved ages, routes and expiries. I constructed two states with identical observations and masks but different hidden-order ages: the same WAIT produced different rewards and next observations. This is direct evidence of state aliasing. Feedforward PPO remains a legitimate controller, but the report must acknowledge its information restriction. See [observation.py, `rank_pending` and `build_observation`](/home/caijihang/projects/medical_uav/medical_uav/env/observation.py).

   The observation decomposition, \(452=80+12\times27+3\times16\), is consistent. Current demand rate and scheduled future intensity are **model knowledge**, not realized-future leakage. They are legitimate only under the stated assumption that the dispatcher knows the load regime and peak schedule. The networks use observation/mask inputs; I found no evidence that PPO accesses future orders.

   Several conclusions must be attributed to the benchmark design:

   - Battery safety is enforced by analytical masks and simulator invariants; it is **not learned safety**.
   - Deadline-aware selection of the 12 visible orders is already a handcrafted scheduling component shared by the agents.
   - A* supplies routes; PPO learns dispatch, not obstacle avoidance or flight control.
   - Non-interruptible charging, deterministic travel, linear charging, fixed handling times, and no ground energy consumption materially shape the charging problem.
   - Episodes drain outstanding work after arrivals stop, without requiring return home or disposal of every expired onboard box. This is a finite-day service objective, not continuous fleet operation.

   These are acceptable simplifications if disclosed. The medical labels and deadlines are benchmark assumptions, not validated clinical utility. See [config.py](/home/caijihang/projects/medical_uav/medical_uav/env/config.py), [masks.py](/home/caijihang/projects/medical_uav/medical_uav/env/masks.py), and [simulator.py, `is_done`](/home/caijihang/projects/medical_uav/medical_uav/env/simulator.py).

   Reward v2 is internally coherent, and complete-day reward accounting reproduced:
   \[
   R=1.1N_{\mathrm{on}}+0.2N_{\mathrm{late}}-0.1N_{\mathrm{released}}-0.001E.
   \]
   It optimizes a compromise involving counts, late service and energy—not exactly mean daily on-time fraction. Equal reward across types also permits sacrificing urgent plasma for easier orders. Reporting daily rates, pooled counts and per-type outcomes is therefore essential. The selection gate limits relative deterioration against Slack; it does not establish clinically appropriate prioritization.

3. **The load design is useful, but §9.18 contains two substantive overclaims.**

   Low/medium/high loads provide a useful ceiling check and two congested regimes. Fixed peaks make anticipation learnable, and the outcomes separate methods without total collapse. Calibration against Slack-aware is acceptable for constructing a benchmark, although it makes this an engineered challenge rather than empirical hospital demand.

   **The claimed capacity argument is not a valid lower bound.** [failure_analysis.py:63](/home/caijihang/projects/medical_uav/scripts/failure_analysis.py:63) adds handling and direct flight separately for every order. Handling is additive for successfully served orders; flight time can be shared by batched cargo. Moreover, orders released within an hour need not finish within that hour, and observed charging share is not an unavoidable hourly allowance. Therefore “65–72/87–94 minutes of work per peak hour,” “high is about 1.5× capacity,” and “some peak orders must fail” do not follow from this calculation. Describe it as an **unbatched workload indicator**, or supply a valid feasibility bound. Do not claim a demonstrated irreducible failure fraction.

   **“No snowball across peaks” is also unsupported.** The analysis regresses afternoon failure rate on morning maximum backlog and afternoon order count, without uncertainty intervals. Morning maximum backlog is not the queue, battery and cargo state entering the afternoon. A small coefficient is neither evidence of equivalence nor evidence of no causal carryover. Furthermore, backlog maxima are sampled only at policy decision times, so long actions can hide larger intervening queues. See [failure_analysis.py:55](/home/caijihang/projects/medical_uav/scripts/failure_analysis.py:55) and [the regression:142](/home/caijihang/projects/medical_uav/scripts/failure_analysis.py:142).

   Replace the conclusion with: **“Queues generally decline between peaks; this exploratory regression did not show a strong association with the chosen morning-backlog summary.”**

4. **The seed and selection discipline is strong; the development record is nevertheless adaptive.**

   Demonstration, online-RL, screen, selection and test ranges are explicitly separated in [splits.py](/home/caijihang/projects/medical_uav/medical_uav/splits.py). Grouping BC validation by demand seed is correct. Selecting one checkpoint per training run across all loads, with low-load and per-type gates, is substantially better than reporting whichever checkpoint wins each column. All saved winners reproduced the implemented rule.

   However:

   - Screen days influenced reward choice, architecture decisions, behavioural hypotheses and later analyses. They are **development evidence**, including when called “confirmation” or “replication.”
   - Selection days are also used for training curves and repeated checkpoint comparisons. Their winning scores are optimistically selected.
   - The control-analysis preregistration explicitly followed inspection of some learning curves. It is an advance analysis commitment relative to later selection/results, not a fully prospective registration before all relevant evidence.
   - The best-of-five shipping rule was adopted after some primary results were known, although before the remaining selections. Disclosure makes this assessable; it does not remove the adaptation.

   These facts do not demonstrate locked-test leakage. They explain why the untouched test remains indispensable. The [control preregistration](/home/caijihang/projects/medical_uav/docs/superpowers/experiments/2026-10-24-m3/codex_preregistration.md) is commendably explicit about what had already been seen.

   Shipping the best eligible seed is legitimate operational selection. It requires counting the five-run search cost and separating the selected controller from typical training performance. The README already does this reasonably well. Seed 11 beating seed 22 by approximately \(3\times10^{-5}\) on selection score does not establish that seed 11 is intrinsically superior.

5. **The statistics answer sensible conditional questions; the controls support narrower conclusions than their labels might suggest.**

   Paired bootstrap over complete days is appropriate. Averaging training seeds within each day avoids treating seed-day rows as independent observations. Recomputing pooled per-type ratios from counts is also correct. See [evaluation.py](/home/caijihang/projects/medical_uav/medical_uav/evaluation.py) and [compare_variants.py](/home/caijihang/projects/medical_uav/scripts/compare_variants.py).

   The important limits are:

   - These intervals describe demand-day uncertainty **conditional on the trained policies**. They do not fully quantify uncertainty over future training runs. Report every seed, SD and range prominently.
   - Three-seed controls provide useful evidence but limited training-population precision, especially for small differences or equivalence.
   - Numerous loads, types, variants and behavioural endpoints create multiple-comparison opportunities. The registered “per-comparison 95%” description is honest, but does not provide simultaneous coverage. Freeze the headline claim hierarchy before test access; treat additional discoveries as exploratory.
   - “No significant difference” is not equivalence. The registered practical margins are useful precisely because they avoid that mistake.

   The reward experiment supports **the whole reward change under this training budget and selection procedure**. It does not isolate the late-reward coefficient: expiry penalties also change, and reward scale can affect optimization.

   The MLP comparison concerns **two complete network designs**. Parameter count, activation, parameter sharing and representation all differ. It supports retaining the set network, not “attention causes the improvement.” See [networks.py](/home/caijihang/projects/medical_uav/medical_uav/agents/networks.py).

   BC→PPO is conditional on one shared BC initialization plus critic warm-up. Its endpoint similarity and inconclusive efficiency result are credible. They do not establish that imitation generally lacks value. The corrected accounting of demonstration and warm-up costs is a strength; the coarse 0.5M evaluation spacing limits early-efficiency conclusions.

6. **The baselines are credible course-project comparators, but they do not exhaust conventional planning.**

   My recomputation from [m3_screen_eval_all.csv](/home/caijihang/projects/medical_uav/runs/m3_screen_eval_all.csv) and [scenario_screen.csv](/home/caijihang/projects/medical_uav/runs/scenario_screen.csv) gives:

   | Development comparison: five-seed PPO mean minus baseline | Medium, percentage points [95% CI] | High |
   |---|---:|---:|
   | Slack-aware | +10.76 [9.07, 12.39] | +16.98 [15.29, 18.66] |
   | Visible-order rollout | +2.01 [0.97, 3.01] | +6.04 [4.78, 7.29] |
   | Scenario rollout, 16 scenarios | +2.07 [1.33, 2.80] | +4.10 [3.07, 5.09] |

   These are substantial gains, especially against Slack. At low load, the planner comparisons do **not** establish superiority; performance is near the ceiling.

   Random, EDF, FCFS and nearest-first are useful sanity references. EDF/FCFS are particularly weak under expiry and charging constraints; beating them contributes little beyond demonstrating basic competence. Slack is more meaningful, and its threshold was calibrated rather than chosen entirely arbitrarily.

   Both planners evaluate all valid first actions, which is a real strength. But subsequent decisions follow Slack-aware. The scenario planner samples arrivals for 60 minutes and then drains modeled work; it does not optimize a complete contingent routing policy. It also omits existing hidden backlog, while PPO receives backlog aggregates. Exact physics and demand-model access help the planner; restricted continuation, horizon and hidden-state treatment limit it. See [rollout.py](/home/caijihang/projects/medical_uav/medical_uav/baselines/rollout.py) and [scenario.py](/home/caijihang/projects/medical_uav/medical_uav/baselines/scenario.py).

   A skeptic can reasonably say: **PPO received millions of offline decisions on the evaluation distribution, while conventional planning received a narrow configuration comparison and a fixed heuristic continuation.** This does not invalidate the benchmark result. It prevents extrapolation to “RL beats planning” or state-of-the-art routing.

   The 16-scenario comparator passes the registered latency bound: I verified all nine load/pass p95 values, with the largest about **762 ms**. Its plasma-gate failure is correctly retained and disclosed under the registered fallback rule. Faster neural inference is established computationally; subsecond planning is already fast relative to minute-scale dispatch actions, so do not portray it as operationally unusable.

7. **My assessment of every §9 evidence item is as follows.**

   Here “strong” means strong **development evidence**, not completed locked-test confirmation. References are to [roadmap §9](/home/caijihang/projects/medical_uav/docs/superpowers/specs/2026-10-23-roadmap-to-final.md:76).

   | Items | Examiner’s assessment |
   |---|---|
   | **1–4: reward, BC/DAgger and scratch-PPO pilots** | Exploratory evidence that justified development choices. Single-seed pilots do not independently confirm the choices they motivated. |
   | **5: production BC** | Strong evidence that high teacher-action accuracy does not guarantee matching closed-loop service. Limited evidence about BC generally because initialization replication is limited. |
   | **6–7: early and final scratch runs** | Useful learning evidence. Update diagnostics support numerical stability, not convergence or optimality. Final checkpoints and subsequently selected checkpoints must remain distinct. |
   | **8: checkpoint selection** | Strong procedural evidence: the gates and winners reproduce. Selected performance is not independent validation. |
   | **9: selected PPO versus visible planner** | Strong development comparison, conditional on the implemented planner. This item uses three PPO seeds; do not mix it with five-seed results. |
   | **10: reward, BC and architecture controls** | Strong v2-versus-M1 result under fixed-budget selection; practical similarity to on-time-only is supported conditionally. Additional served/plasma gains are useful secondary evidence. No BC efficiency benefit is established. Architecture causation remains unisolated. |
   | **11: shipped controller** | Selection and identity are well supported. Its best-seed performance is not representative of a randomly trained agent. |
   | **12 and 17: charging, hub use and distance** | Strong descriptive associations. Similar boxes per stop undermine a “better batching” headline, but do not prove batching contributes nothing. Hub use and reduced pickup travel are plausible mechanisms, not isolated causes. |
   | **13: energy intervention** | Strong evidence about the specified interventions. No evidence that early charging alone explains PPO’s advantage, or that masking identifies a mediated fraction. |
   | **14 and 16: scenario rollout** | Strong recorded development performance and latency evidence. Sampling noise as the reason four scenarios perform poorly remains a hypothesis. Plasma sacrifice must remain visible. |
   | **15: unseen-map stress** | Useful zero-shot transfer evidence within this map generator. PPO retains a rule-baseline advantage, but is slightly behind the visible planner at medium load. |
   | **18: failure analysis** | Outcome breakdowns are informative. The capacity/inevitability and “no snowball” conclusions overreach for the reasons above. |

   The strongest final narrative is therefore **improved timely dispatch, reward trade-offs, and coordinated use of the hub and charging opportunities**—not a predetermined batching success story.

8. **The energy and map studies improve the project, provided their causal and generalization boundaries remain explicit.**

   The energy study has a proper full-charge control, fixed intervention parameters, manipulation checks and all-five-seed masking. I reproduced the selection-day primary result: **+0.03 points [−0.46, +0.51]**, despite increased peak-start battery. Removing full-charge actions reduces PPO performance by **2.32 points [2.10, 2.54]** on the combined endpoint. These results reject a simplistic “higher battery reproduces PPO” account. They do not identify optimal charging timing or how much of PPO’s advantage flows through battery. The [final-phase preregistration, item 1](/home/caijihang/projects/medical_uav/docs/superpowers/specs/2026-10-27-final-phase-preregistration.md) handles this distinction well.

   Hub behaviour is particularly distribution-specific: **80% of order types by probability touch M**, and M also hosts a charger. Learning to exploit that structure is a valid achievement, but a conventional hub-aware rule could plausibly capture part of it.

   The map stress test appropriately fixes maps and avoids reselection. Its map bootstrap is conditional on the reused demand days and trained policies; ten maps give limited precision. Use the existing map-0 results on the **same 20 demand days** when discussing transfer loss. Against the planner, that matched reference is approximately **+2.05/+5.69 points**, versus **−0.58/+1.98** on unseen maps. Roadmap §9.15 instead juxtaposes some screen-day reference numbers.

   The later [multi-map preregistration](/home/caijihang/projects/medical_uav/docs/superpowers/specs/2026-10-27-multimap-preregistration.md) explicitly records the owner-requested scope expansion, and runs exist. Reconcile it with the earlier “no new campaign” instruction and 33-configuration manifest. Keep the original single-map result intact; do not silently replace it with whichever generalization result looks best.

9. **Implementation quality is good, but the final audit should close several specific gaps.**

   I found no substantive defect in the inspected PPO/GAE implementation: stored masks and log-probabilities, terminal boundaries, reward scaling, clipped objective, effective \(0.25\,\mathrm{MSE}\) value coefficient and policy-preserving critic warm-up are consistent. With an undiscounted finite-day objective, \(\gamma=1\) is appropriate despite variable action duration; GAE’s per-decision \(\lambda\) is an estimator choice. See [ppo.py](/home/caijihang/projects/medical_uav/medical_uav/agents/ppo.py) and [test_ppo.py](/home/caijihang/projects/medical_uav/tests/test_ppo.py).

   The planner tests provide meaningful evidence against hidden/future-order leakage. Nevertheless, **registration and implementation are not identical**:

   - Full-precision simulator fields replace the registered observation-decoding interface; this deviation is documented.
   - Scenario RNG initialization uses `[master_seed, decision_index, scenario_index]`, resetting the decision index each episode. The registered episode-counter component is absent. This creates a deterministic reused random-stream scheme, not the registered independent episode streams. It is not future leakage, but should be recorded explicitly before test freeze.
   - The 100-state mutation test does not itself establish a strict equal-recomputed-observation contract: hidden counts/destinations are changed, and the view is not recomputed. The separate restricted-view test is valuable, but narrower than every promised audit condition.

   See [scenario.py:115](/home/caijihang/projects/medical_uav/medical_uav/baselines/scenario.py:115) and [test_scenario.py](/home/caijihang/projects/medical_uav/tests/test_scenario.py). Resolve these through an explicit deviation record or a justified pre-test correction with regenerated affected development evidence.

   The replay reconstructs physical state and outcomes from event history and records network choices before actions—good foundations. However, its pending table displays the **full analyst backlog**, sorted by deadline and reordered for highlighting, rather than the policy’s actual 12 ranked slots. Label that distinction. Also label displayed probabilities as decision-time policy probabilities; greedy execution is not sampling, and probabilities do not explain causal reasoning. See [render.py:279](/home/caijihang/projects/medical_uav/medical_uav/env/render.py:279) and [demo.py](/home/caijihang/projects/medical_uav/scripts/demo.py).

10. **Before 20 November, prioritize completion and auditability in this order.**

   | Rank | Required work | Value / approximate cost |
   |---|---|---|
   | **1** | Freeze a complete manifest and build a resumable, atomic locked-test runner. Include all methods, checkpoint hashes, configurations, analysis versions, counts and intervention outputs. Dry-run reporting on development data first. Reconcile multi-map entrants. | Essential; moderate engineering, existing evaluation budget. |
   | **2** | Correct the partial-observability formulation, capacity argument, snowball wording, and causal/architecture claims. Record remaining planner deviations. | Very high value; hours to one day. |
   | **3** | Export portable evidence: selected checkpoints for every reported learned entrant, per-day CSVs, selection records, preregistrations, hashes and reproduction commands. Rehearse the actual submission archive and installation route. | Essential; roughly one day plus rehearsal. |
   | **4** | Execute the frozen locked campaign once, then report per-load paired effects, margins, per-type counts, all seed results and the shipped model separately. Preserve failures and all registered contrasts. | Highest evidential payoff; compute cost should follow measured planner timing. |
   | **5** | Complete the registered median/worst/plasma-failure cases and synchronized comparison video. Use event-derived backlog traces and clearly distinguish agent-visible information from analyst information. | High explanatory value; one–two days. |
   | **6** | Finish only the already registered multi-map comparison, without further architecture or budget changes. Keep it secondary if packaging or the locked campaign is threatened. | Useful extension; bounded by the existing commitment. |

   Reproducibility is currently strongest for the shipped model, whose [manifest and checksum](/home/caijihang/projects/medical_uav/checkpoints/ppo_set.json) check out. It is weaker for the entire evidence package: `runs/` and `outputs/` are gitignored, while [.gitattributes](/home/caijihang/projects/medical_uav/.gitattributes) excludes protocols and tests from archives. The advertised root M3 report is also absent. Existing checkpoints are inference snapshots, not exact training-resumption checkpoints; say so. These are packaging/provenance deficiencies, not reasons to retrain.

   **Do not** tune on locked results, extend the winning seed, change reward or load definitions to improve the story, revive planner imitation/DAgger, add another architecture, or build a hidden-backlog belief model now. A new hub-aware heuristic or planner sweep could be interesting, but is unnecessary if the final claims remain explicitly limited to the frozen comparators. Reserve time for each student to explain and independently reproduce the work; AI review records do not substitute for that understanding.

11. **Overall verdict:** **B+ for rigor in its present state, with a credible route to A−/A-level course-project work after locked evaluation and packaging.** The replicated gains, controlled comparisons, explicit selection and negative intervention result are substantial strengths. The weaknesses are adaptive development evidence, limited planner/generalization scope, two unsupported failure-analysis conclusions, and incomplete portable provenance. I would regard this as a strong synthetic dispatch study—not yet a confirmed general advantage of PPO over planning, and not evidence of clinically valid UAV operations.
