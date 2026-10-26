# Discussion publication and review

Original supplied documents/checksums remain in `discussion_plan_source/`. Current decisions and evidence are in `discussion_plan_review.md` and `results_index.md`.

1. Base: `d050c4c100b86bdb6676d57f400a24c9bc80e66b`; branch `release/discussion-v0.1`. Remote default was read as `main`. Preserve remote settings, history and tags; no force push.
2. Inspect staged files, history blob sizes and credential patterns before pushing. Stage explicit release paths. Archives contain hash-verified selected evidence, no environments or credential files.
3. In a new checkout, install the README CPU environment; verify artifacts, run the PPO demo, replay a full Dev day, recompute tables and run tests. Confirm imports resolve inside that checkout. Separate author checks from independent teammate review and remote CI.
4. Push this named branch only. Create a Draft PR to `main`, title `Baseline discussion release: destination-conditioned PPO and Gen4 roadmap`, using `RELEASE_DISCUSSION.md` and the actual receipt. Do not automatically mark ready or merge.
5. Another member clones the remote branch and records commit, versions and results on the PR. Author-created fresh clones do not replace independent review.
6. Only after review, create a new annotated tag such as `v0.1.0-discussion.1` on the approved commit, push that exact tag and create a prerelease. Do not move an old tag or push all tags. This gate comes directly from the supplied plan.
7. Register research hypotheses on independent branches. Schema changes need mapping/mask/baseline acceptance. Historical Tests are not new-method selection data; new Final-Test is not ordinary CI.

The connector returned 404 for this private repository while local SSH and the logged-in browser could read it. This access mismatch does not show that the user lacks permissions. Never print or commit tokens.
