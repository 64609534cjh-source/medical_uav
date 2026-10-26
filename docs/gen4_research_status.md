# Gen4 research checkpoint — R1 completed, bounded planning next

The discussion baseline stays frozen. [PR #1](https://github.com/64609534cjh-source/medical_uav/pull/1) is Draft. The repaired [remote CPU check](https://github.com/64609534cjh-source/medical_uav/actions/runs/40351344438) passed at `a9117e3`; independent teammate reproduction/review is still pending. No tag, prerelease or merge is inferred from this check.

## R1 evidence

New diagnostic source `d21f5992b32f50dffaea5c50f9ee65b2425a628a` replayed ten frozen selected models over 3,000 legacy Dev-extra days. Every pre-existing serialized day field and failure diagnostic remained exact. No optimizer update or Test evaluation was performed. Summary: `results/gen4_r1_summary.md`; all original order/day records: `artifacts/gen4_r1_visibility.tar.gz`, with hash/member index in `artifacts/gen4_r1_visibility_manifest.json`. This archive adds research evidence without replacing the original discussion archives.

Main observations:

- No failed order was already direct-late on its **first visible decision** in this replay. Almost every failed order had at least one legal PICK opportunity. These facts weaken a simple late-first-visibility explanation; they do not establish that all failures were avoidable under shared capacity/time/energy constraints.
- At High load, 19.49%/35.20%/25.71%/29.22% of destination failures (plasma/medicine/sample/supplies) were hidden at some decision. Visibility can be lost after an earlier opportunity. These conditional, repeated-order percentages are not the policy's overall failure rates or causal truncation effects.
- Decision-held hidden waiting accounts for 3.26% of all destination High pending-wait minutes; observed-visible waiting is 89.44%, with 7.30% unobserved between decisions. Labels are not continuously recomputed inside macro actions.
- Same-state queries expose consistent pickup/delivery preference differences around urgent cargo. Those action frequencies and four representative divergence cases explain behavior, not counterfactual outcomes.

The mechanism evidence now favors a bounded public-information sequencing/lookahead feasibility test before committing large training budgets to K=24. Keep the candidate-window interface/control experiment registered; the observed hidden fraction alone does not predict its service gain.

## Development registration

Audited 677 JSON/JSONL records, 39,265 unique explicitly logged demand IDs, learner IDs 0–5 and formal map 0. Detailed file hashes and observed IDs are compressed in `results/gen4_seed_usage_audit.json.gz`; the readable summary states the audit scope. This is a local-record audit, not a claim about unrecorded external experiments.

`configs/gen4_scenario_registry.json` reserves:

- Dev-tune 100140–100239, all three loads, 100 days.
- Dev-confirm 100240–100339, separately registered development confirmation, not Test.
- Explore learners 11/12, long-screen 13/14/15, final confirmation 21–25.
- Train remains 0–99999; all three old Tests remain consumed and excluded from selection/training.

At completion of R1, neither new development list has been evaluated. Any later use is logged in a separate access record so the original registration remains immutable. New Final-Test is still null/uncreated; final primary candidate and validation protocol must be frozen first. Existing trainer's historical 40-day selection is not silently claimed to implement this new 100-day protocol.

## Next registered work

1. `docs/experiments/GEN4_R4_LOOKAHEAD.md`: validate a public-only state adapter, independent absolute-time scenarios, live-state immutability, legal fallbacks and complete-day compute before performance claims.
2. `docs/experiments/GEN4_R3_CANDIDATE_WINDOW.md`: K=12 compatibility, K=24 schema/order/mask controls and equally adapted rules before any learning run.

The ~0.5pp promotion value remains a proposal for the eventual learning screen, not a retrospective significance cutoff. Keep the user's original overall on-time/reward selection and report all type tradeoffs; do not introduce new service constraints. Budgets remain maxima, and no new multi-million-transition suite is authorized merely by a positive diagnostic pattern.

Reproduce R1 after `python scripts/discussion.py verify` by extracting the indexed R1 archive, then running `python scripts/analyze_r1_visibility.py`. The raw archive contains the original ignored `outputs/...` paths. To rerun simulation, check out the frozen diagnostic source, use a fresh output location and the registered protocol; do not overwrite evidence. The existing source archive extractor only handles the discussion archives; the R1 archive is separately indexed and should be extracted only after validating its SHA256 and member paths.

Codex assisted implementation, read-only replay, registration and reporting; the group is responsible for independent review and final conclusions.
