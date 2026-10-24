**Recommendation: finish and submit M2; keep the set network primary; prioritise valid comparisons and explanations over more architectures.** The PPO CSVs support the reported screen results. They do not yet establish selected-checkpoint performance, convergence, or superiority over a validated planner.

I reviewed the specified documents, branch code/diffs, archive contents and result CSVs. I did not rerun training/tests; the default Python lacks PyTorch. Installation/test evidence therefore comes from the execution ledger.

**1. Before M2 submission**

I found **no new blocking algorithmic defect** in the reviewed M2 implementation. The branch contains commented networks, BC training/checkpoints, forward/backward demonstration, graphical demos, evaluator, README and pinned dependencies. PPO is correctly excluded. The previous substantive fixes are present.

The remaining submission requirements are:

- **Add the actual two-page report.** The README references it, but `git archive m2-network` currently contains **no PDF**. `docs/` is export-ignored, so explicitly insert the report into the final ZIP.
- **Check the final ZIP against the README’s exact installation route.** The ledger documents fresh Conda Python plus pip installation; that does not itself demonstrate `conda env create -f environment.yml`. One exact-command rehearsal suffices. Use the approved M2 branch, not the older plan’s instruction to merge into `main`.
- **Make claims match the delivered artifact.** BC learns the teacher’s actions; it is not PPO and does not train the value head. The separate backward-pass demo exercises the critic but does not make the shipped BC critic trained. Preliminary PPO belongs only in a clearly labelled outlook sentence.

One small traceability repair is worthwhile: `bc_history.json` lacks environment/version metadata despite the roadmap’s requirement. Attach the version, training configuration and source revision to the history or a manifest. This does **not** justify retraining.

No further video redesign is needed for M2 given the owner’s acceptance.

**2. What the two pages should contain**

There is no evidence explaining the missing four M1 marks beyond the GIF comment. Do not assume additional experiments alone will improve the score. Make the implementation and your understanding easier to assess.

| Space | Content |
|---|---|
| Page 1, roughly ⅔ | One architecture diagram: `452 = 80 + 12×27 + 3×16`; separate encoders; global-query cross-attention; shared PICK scorer; 14 other logits; value head; action masking. Give both parameter counts: set **62,928**, MLP **370,459**. Explain the empty-slot mask and null token. |
| Page 1, remainder | The short execution chain: collect teacher episodes → seed-separated BC training → checkpoint → greedy policy → evaluator. Describe the forward/backward demo, including numerical outputs, gradients and optimizer step. |
| Page 2, roughly ½ | Small train/validation loss plot and compact closed-loop results table. Include Slack-aware and both BC networks; other baseline rows if readable. State 900 collected episodes, **720 training / 180 validation**, grouped by demand seed, best validation-loss checkpoint, and 40 screen days per load. |
| Page 2, remainder | Libraries/external tools and accurate Claude/Codex disclosure; substantive reflections; a readable explanation-frame thumbnail if space permits. |

The strongest reflection is already supported: **93.5–93.9% teacher-action accuracy does not reproduce teacher performance in closed loop.** BC-set reaches 79.5%/64.4% at medium/high versus Slack-aware’s 81.0%/65.9%. This motivates PPO without presenting imitation as a success beyond the evidence.

Explicitly disclose the reward change **including expiry penalty 0.5→0.1**, observation 403→452, separate cargo/order encoders, fixed non-PICK heads, and fixed-map scope. Attribute reward selection to preliminary pilot evidence.

Leave out repeated M1 mechanics, code listings, a large PPO results section, and claims that attention causes the improvement. The current network uses **cross-attention**, not order-to-order self-attention. BC validation contains no RETURN examples; do not imply every action family was validated.

**3. Ordered priorities and budgets**

Budgets below are caps on additional work, not reasons to expand scope.

1. **Finish M2 report/package — ½–1 person-day; zero training.**  
   Freeze this before extending M3.

2. **Implement formal selection and result provenance — 1 person-day; zero training.**  
   Evaluate the existing 12 checkpoints per run on all 100 selection days and all three loads: about **21,600 episodes** across six runs. Fix the per-type aggregation convention and tie-break before selection. Current CSVs contain per-type rates but **not denominators**; add per-type released/on-time counts. Save selected paths, eligibility checks, configurations and revisions. The existing 20-day curves cannot enforce the roadmap’s low-load constraint.

3. **Audit and productionise the visible-order planner — 1–2 person-days.**  
   Run the 100-state information audit before generating demonstrations. Perturb hidden/future details while holding the policy’s allowed information fixed. Note that PPO legitimately sees aggregate backlog, demand rate and scheduled demand intensity; “unknown future orders” does not mean “no forecast information.” Measure planner latency and re-evaluate it under v2. The old spike imports removed `DEV_SEEDS` and is historical evidence, not a runnable production baseline.

4. **Run the useful controls — approximately 2 person-days implementation plus 54M decisions.**  
   Slack-BC→PPO: **3×6M**; M1 reward and on-time-only: **2×3×6M**. Keep architecture, observation, dynamics and PPO settings fixed. Restore the actual M1 late-reward function, not merely its expiry coefficient. For BC warm-up, freeze the actor’s **shared encoders as well as policy heads**; verify policy logits stay unchanged. Report demonstration and warm-up costs separately. Treat “objective design matters” as a hypothesis to confirm, not a predetermined conclusion.

5. **Behaviour analysis and M3 delivery — 1–2 person-days; zero new training.**  
   Prioritise aggregate batching, charging timing, peak/off-peak service and per-type outcomes, then two matched-day examples and one failure. Use a few feasible-state probes; probabilities/attention weights alone do not establish causal explanations. Ship a selected model, simple testing command, training code and two-page report by 6 November. Move unfinished controls to final rather than jeopardising packaging.

6. **Final conventional comparison and seed replication — 2–3 person-days plus 12M decisions.**  
   Add primary seeds 44/55 at **6M each**. Build scenario rollout with the roadmap’s bounded **4 versus 16 scenarios, 60-minute horizon, p95 ≤1 s** protocol. Measure latency on the same hardware. Establish its connection to a specific published conventional method before claiming the bonus; “scenario rollout” alone does not establish state-of-the-art status.

7. **Planner-BC→PPO — conditional, 18M decisions plus demonstration cost.**  
   Meaningful for sample efficiency or distillation, but lower priority now that scratch PPO already exceeds the planner spike. Proceed after the audit and only if the teacher supplies a useful behavioural contrast. Keep BC-only results and early learning curves.

8. **Freeze methods, run locked test, produce final artifacts — reserve 3 person-days.**  
   Evaluate the frozen selected models and baselines on 300 test days; bootstrap paired whole days and report training-seed spread separately. Produce the synchronised comparison video and independently written individual reports. Keep the roadmap’s final buffer.

9. **Optional extensions only after those commitments.**  
   - **Longer training:** cap at three additional 6M continuations of the primary agent. Current checkpoints save weights, not optimizer/RNG/collector state: loading them is not exact resumption. Define and record the continuation schedule; keep the 6M comparison intact. A 12M-from-scratch schedule is a different experiment.
   - **Transformer:** defer. Existing set-versus-MLP results already provide an architecture comparison, but differ in parameter count, activation and actor/critic sharing. A controlled pooling-versus-attention ablation is cheaper and more interpretable if attention is the question.
   - **Clairvoyant planner:** drop from the core plan. A heuristic with future information is **not a mathematical upper bound**. Call it a privileged-information reference unless an exact solution or certified bound is available.

**4. Main final-grade risks and cheapest guards**

- **Scope outruns explanation:** reserve report/video time now; require each member to explain masking, BC versus PPO, reward design and one failure.
- **Inflated claims:** distinguish screen/select/test; report BC shortcomings; call the current architecture comparison a comparison of complete designs; avoid “optimal,” “upper bound” and “state of the art” without evidence.
- **Generalisation promised but untested:** either explicitly narrow the final claim to unseen demand on map 0, or add a small separately labelled unseen-map stress test. Adding travel times to the observation does not demonstrate map transfer.
- **Untraceable experiments:** record reward configuration, source revision, seed ranges, selected checkpoint and file hashes. Current PPO logs lack version metadata, and actual training episode seeds are not retained.
- **Submission failure:** rehearse the actual ZIP, verify the report is included, disclose AI assistance accurately, and keep individual reports independently written.

The best return now is **credible selection, a strong audited conventional comparator, and clear behavioural evidence**. Another network or an unaudited oracle would add less.