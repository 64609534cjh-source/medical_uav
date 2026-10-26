# GEN4-R1-VISIBILITY — waiting and opportunity diagnosis

Status: registered diagnostic replay, no training. Branch `research/failure-audit`, based on publication repair `a9117e3`. Implementation/run owner: Codex; independent human reviewer not assigned. Protocol: `configs/gen4_r1_protocol.json`.

Question: do remaining failures accompany late first visibility, actual candidate exclusion, or repeated unchosen legal pickup opportunities? These are competing descriptive mechanisms, not identified causal effects or avoidable-loss bounds.

Frozen M5 and destination best models, five paired learners 1–5, 100 legacy Dev-extra days 100040–100139 × Low/Medium/High. Existing model hashes and training source SHAs come from `artifacts/checkpoint_manifest.json`. No checkpoint reselection, reward change, network change, optimizer update, new development confirmation or Test evaluation.

The observer records all released orders, mutually exclusive final outcomes, first-visible physical delay/direct slack, counts of visible/hidden decision states, legal PICK opportunities, and waiting duration. Macro actions have no intermediate policy decision: a visibility label is held only as a **decision-held** descriptor until the next decision or pickup/expiry. Arrivals before the first observed decision remain a separate unobserved-between-decisions component. This is not a continuous-time candidate ranking and does not classify every never-visible order as truncation.

Conditional action groups are fixed before replay: overflow (>12 pending), urgent live cargo (deadline within 15 min), battery below 40, and their complements. On each visited state, query the paired other method with exactly the same 403 observation and 26-action mask. Queries measure action disagreement on a common state, not the other method's counterfactual reward. Groups overlap; their rows cannot be summed as disjoint error attribution.

Acceptance: all 3,000 original serialized day records and original diagnostics remain exactly equal after removing the new `visibility_audit`; all actions legal, all complete days, per-order waiting components sum to pending duration. Synthetic tests distinguish orders born and expired inside charging from top-K exclusion. Failed runs are retained and never included as complete.

Outputs: compressed order/day data, grouped summary, representative trace comparisons, source/protocol/model hashes and wall time. Summaries report day/type rates and pooled conditional order counts separately. Five learner repetitions do not create 500 independent demand IDs. Descriptive intervals are not interpreted as causal confidence bounds. Existing paired service intervals remain in Gen3 reports.

This diagnostic consumes zero training transitions and zero planning simulations. Only the replay wall time and 3,000 existing-development episodes count toward diagnostic compute. It does not spend the 8×2 training search allowance or unlock new confirmation data.
