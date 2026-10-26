# Gen4 checkpoint — R1 evidence and two bounded planning probes

Recorded 2026-10-27, Asia/Singapore. Formal method remains the frozen destination network plus its guarded fast inference path. This is progress in the registered research line, not a new mainline architecture or a final experimental conclusion.

## What is complete

- Discussion release [Draft PR #1](https://github.com/64609534cjh-source/medical_uav/pull/1): remote 200-test/Dev reproduction check passed at `a9117e3`. Independent group-member review remains pending; no release tag is created.
- R1 [Draft PR #2](https://github.com/64609534cjh-source/medical_uav/pull/2): 3,000 original Dev-extra days are exactly preserved, new order waiting/opportunity and same-state action diagnostics are archived, separate tune/confirmation lists are registered. Its remote [203-test check](https://github.com/64609534cjh-source/medical_uav/actions/runs/40355024845) passed.
- Public-information lookahead prototype: no live simulator/day-seed input; independent shared absolute-time synthetic futures; full macro-action settlement; terminal value zero; complete comparison or exact frozen-action fallback. Tests cover poisoned real future/day ID, live order immutability, hidden-history rejection, zero-budget fallback and full-copy/fast-copy score parity.

## Engineering result

Two implementations each evaluated reference/no-value/value-tail on the same preselected five Dev-tune days × three loads, frozen learner 1. M=4, H=30 simulated minutes, 20ms soft budget and candidate rule were unchanged. v1 source `9c09f3312c1190db85d6a3a3ca65a877aa1d1f3e`; v2 source `20cb5b9` (full SHA in raw manifest).

Every complete-day on-time rate matched the reference; the five-day equal score is 87.79%, not a Test result. v2 finished only 19/1,765 comparisons (1.08%) and changed four actions. p95 latency remained about 20.4ms versus 0.27ms reference; maximum observed overshoot was 46.26ms. The static/dynamic clone optimization passed exact unbounded score tests but did not materially resolve fixed-budget coverage. Do not promote this prototype or infer that planning generally cannot help.

The two bottlenecks are separate: 695/1,765 states lacked enough public recent-arrival history, and about 553 eligible decisions exceeded budget. There were also 466 no-work and 32 hidden-order fallbacks. Completed/late/expired total service and all type rates remain inspectable in raw day data; these short probes are not a full five-learner service study.

Recorded feasibility-evaluation cost: 90 complete episodes including repeated controls, five unique demand IDs, one frozen learner, **96,366 extra planning simulator transitions**, zero learning transitions. Unit tests and the single Train-state micro-profile are separate engineering checks excluded from that simulator count. The two null results and all timing outliers are retained.

## Data access and next steps

Only Dev-tune 100140–100144 was used. The remaining 95 tune days and all 100 confirmation days remain unevaluated; new Final-Test is still absent. `results/gen4_development_access.json` records the two accesses separately from the immutable original registration.

Next priority is a registered batch-of-scenarios scoring implementation with scalar-score parity checks. A planner that retains public observation history is a separate information/state variant and needs its own contract; do not fill missing arrival history by inspecting the live simulator. Candidate K=24 remains a registered bounded interface control, not an established solution to the diagnosed failures. No additional long training is justified by these feasibility scores alone.

## Reproduce archived analysis

From an installed checkout of this research branch:

```bash
python scripts/discussion.py verify
python scripts/extract_research_evidence.py
python scripts/analyze_r1_visibility.py
python scripts/analyze_lookahead_feasibility.py
python -m pytest -q
```

Archive extraction checks whole-file/member hashes and refuses changed local evidence or paths outside `outputs`. The two analysis commands recompute from archived records without running a policy or reopening a Test. Source metadata records the analysis commit actually used. To repeat timing, check out the relevant frozen source/protocol, use a fresh output directory and run each version sequentially; wall-clock decisions are hardware/load dependent, so timing-dependent action equality across machines is not promised.

Full tables and inspected export: `results/gen4_lookahead_feasibility.md`, `.json`, `.png`, `.svg`. Keep these engineering observations separate from historical cross-learner Test confidence intervals. Codex assisted implementation, execution, analysis and documentation; the group remains responsible for review and conclusions.
