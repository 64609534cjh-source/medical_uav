# Author reproduction receipt

Tested remote commit: `c8aa7bd02771c0b7b33ce73163cf42300c3b4a69` on `release/discussion-v0.1`.

A real SSH clone into a new directory used a separate virtual environment and verified that imports resolve in the new checkout. It reused the existing pinned CPU dependency installation through system-site-packages; this is not a clean OS/conda installation or independent teammate review. Python and every package in requirements-lock.txt matched exactly.

- 35 selected checkpoints and 288 archived evidence files verified and extracted.
- Original network demo: 256 real Train transitions, Actor/Critic backward update, eight changed parameter tensors, exact checkpoint reload.
- Default destination learner 1 / Dev 100000 / High: complete day, zero illegal actions, exact original-forward and archived training-time Dev day match; zero near-tie fallback states on this day. Forced fallback remains covered by unit tests.
- All three historical tables recomputed separately from saved days; no Test policy rerun.
- Full test suite: **200 passed in 31.89s**.
- Checkout remained clean after all checks. Author clone/acceptance took 127.58 seconds including network transfer.
- Prior 22 protected source files and 98 prior results/config files preserve their hashes.

No new training experiment or holdout evaluation was run. Eight illustrative trajectories replayed previously used Dev-extra days only. Existing CPU timing/4,800-day equivalence records remain historical evidence, not newly repeated benchmarks.

Git history had no blob above 50MiB before packaging. New model/evidence blobs are 35,390,815 / 52,033,409 bytes (each below 50MiB); scanned credential patterns had no matches. Pattern scanning is limited, not a guarantee against every secret. Four git-diff whitespace warnings in the copied roadmap/source roadmap are intentional Markdown two-space line breaks; original source checksums are preserved.

Initial GitHub CI run 40350244506 installed the pinned environment successfully and passed 199 of 200 tests. One test required bitwise equality between an eight-output head and the corresponding rows of a 14-output head; the remote CPU produced a 3.725290298461914e-9 difference in one output. The release repair keeps weights/bias rows, PICK outputs and critic outputs exact, and uses atol=1e-7/rtol=1e-6 only for the differently shaped head computation. No network implementation, model tensor, environment, PPO update or historical result changed. Remote CI after the repair is reported separately; the original author receipt remains an accurate record of the earlier local run.

Another group member must still reproduce from the remote and review before any new release tag/prerelease. The authored PR body is `docs/RELEASE_DISCUSSION.md`.
