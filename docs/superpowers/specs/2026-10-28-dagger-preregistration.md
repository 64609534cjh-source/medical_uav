# Visible-horizon DAgger — preregistration (2026-10-28)

Written and committed before collecting any DAgger data, including the smoke test.
This exploratory round follows roadmap §9.21/§9.23 and the improvement-round registration.
The historical fresh-day medium/high gaps to the visible horizon are −1.12/+0.32 points.
The objective is to improve medium on-time while retaining expiry05's expiry and high-load
performance, then improve served. Imitation will likely finish slightly below its teacher;
neither teacher matching nor any owner target is promised. Deployment remains one masked
network forward pass, with no planner, ensemble, or test-time adaptation.

## Fixed design and training budget

Five independent students, seeds 11,22,33,44,55, start from the corresponding selected
`paths.expiry05` checkpoints in `runs/improvement_choice.json`. Preserve the architecture,
observation dimension and full travel table. Record the exact initial paths and file hashes.
Teacher: the existing `RollingHorizon(depth=3, foresight=False)`, unchanged tie breaks and
default **v2 planning reward**, as used by the comparator. Rollout physics/observations are
unchanged; imitation uses no rewards. Do not silently change the teacher to expiry05 reward.

Run exactly **three iterations, 16 full medium days per iteration per student** (240 medium
student-days total). The current student acts greedily at every decision, including WAIT;
there is no teacher execution or teacher/student action mixture. Before each executed action,
save `planning_root(env)`: copied visible slots with original IDs/order, cargo, time, place,
battery, map and settings, with no hidden/future arrivals. Require its action mask to equal
the student's recorded mask. Label that saved root with the visible horizon in a process pool;
never reset a teacher environment to merely the same day and pretend its state is the same.
Keep executed `action` and teacher `label` separately, and validate both against that mask.

Choose **fixed low/high self-labelled replay**, rather than PPO fine-tuning, for retention.
Before iteration 1, collect 16 low and 16 high full days using the initial student; its greedy
executed actions are also its labels. Keep this replay fixed in all three iterations. Its
single-forward-pass labels cost milliseconds instead of high-load horizon queries (~3.2 s
each), and avoid another PPO collection, critic warm-up and optimization phase. This is a
cost choice, not a claim of guaranteed retention. The selection gates below measure retention.

Aggregate every medium iteration with all earlier medium records and the fixed replay, with
equal weight per decision. After each collection, train five full shuffled epochs on the
entire aggregate, masked cross-entropy to `label`, Adam LR 3e-5, batch 256, gradient norm
clip 0.5. Reset Adam per iteration, use training seed + iteration for minibatch shuffling.
No entropy, value loss, early stopping, validation-driven epoch choice or PPO phase.
Save the initial checkpoint (iteration 0) and each of the three iteration checkpoints,
the cumulative dataset after each iteration, counts, seed schedule, timing and losses.
No teacher queries outside medium load, no extra iteration, retries, seed replacement or
hyperparameter search after observing selection or confirmation performance.

## Disjoint days and smoke isolation

Add `DAGGER_SEEDS = range(700_000, 710_000)` and
`CONFIRM2_SEEDS = range(600_000, 600_300)` to `medical_uav/splits.py`.
These are disjoint from demonstrations, RL, screen, selection, locked test, stress/energy
days 300000–300099/400000–400099, and historical fresh days 500000–500299.
For student index k=0..4 in seed order, use base=700000+1000*k: medium days base..base+47
(consecutive 16-day iteration blocks), replay days base+500..base+515 at both low/high.
No random reuse of these blocks across students. Map 0 throughout.

Smoke uses only the reserved block 709900–709999, one iteration, one medium day and one
replay day per load, one epoch, release window 20 minutes and two label workers. It checks
plumbing, aggregation, reload and existing evaluator/selector compatibility, never performance.
Synthetic/unit-test episodes use that reserved block or existing debugging/training blocks,
never CONFIRM2. Smoke output is separate from campaign checkpoints and selection.

## Selection, frozen before confirmation

Evaluate all four checkpoints of every student and its matching expiry05 initial checkpoint
on all 100 SELECT_SEEDS, all three loads, map 0, greedily. For each training seed, eligibility:

- High on-time >= matched expiry05 high on-time −0.005 (0.5 percentage points).
- Equal-load mean medium/high expired <= matched expiry05 mean expired +0.003 (0.3 points).

Among eligible checkpoints maximize medium on-time; ties prefer lower medium/high expiry,
then higher medium/high served, then earlier iteration. Served=1−expired at episode completion,
so the served tie break is normally redundant. Iteration 0 is eligible by construction; if it
wins report no selected adaptation for that seed. All five seeds remain in the analysis.
Non-finite or incomplete results, crashes or invalid labels stop the campaign; no fallback
for missing data. Freeze selected paths, initial paths, hashes and all gate results in
`runs/dagger_choice.json` before reading or generating any confirmation outcome.

## New confirmation and reporting

Only CONFIRM2_SEEDS 600000–600299, never previously used, all loads, map 0. Re-run the selected
five DAgger students, five matched expiry05 checkpoints, five frozen v2 entrants, Slack-aware,
visible-order rollout and scenario16. Re-run visible horizon (foresight=False, default v2)
on medium/high on these **new** days; never reuse historical fresh horizon rows. Deterministic
references repeat the same day across student seeds, without invented baseline seeds.

Reuse `docs/superpowers/experiments/2026-10-27-final/improvement_analysis.py` count audits,
estimators, paired-day and independent matched-seed × day bootstrap, and owner-target reading
code. Parameterize its confirmation days/reference set rather than copying its statistics.
Report on-time, served, expired, per-load and medium/high equal-load means, each seed, each
order type, absolute rates and candidate-minus-reference differences. Overall rates average
daily fractions then seeds; per-type rates pool numerator/denominator on resampling. Require
exactly matched days/released counts, finite data, served+expired=1 and coherent type counts.
Use 10000 percentile draws, RNG 0, paired whole days and two-way training seeds × days, 95% CIs.

Print PASS/FAIL for the same three owner targets, on the medium/high overall endpoint, with
**both** interval lower bounds meeting the condition:

1. On-time superiority to every reference (expiry05, frozen v2, horizon, scenario16, rollout,
   Slack): lower bound >0. Report served superiority separately for each reference.
2. Horizon non-inferiority within the explicit 0.5-point tolerance: lower bound >−0.005.
3. Frozen v2 expiry reduction point estimate >=0.01 and lower bound >0.

Also report candidate−expiry05 expiry and high on-time intervals, and whether confirmation
retains the selection tolerances; report literal on-time >= horizon point estimates per load
separately from the registered non-inferiority reading. Expiry no worse than expiry05 is an
owner objective, not established by passing a +0.3-point selection allowance. Failed targets
remain failures, without tuning on confirmation. Write `runs/dagger_report.txt/json`, and only
after complete audited reporting write `runs/dagger.done`; a target FAIL still completes a run.

## Machine budget and authorized execution

At most 24 single-thread CPU workers total; process-parallel teacher queries over saved
student states, and day-parallel confirmation. Train the five students sequentially with
up to 24 label workers each, then selection, then baselines and horizon concurrently with
their worker totals <=24. Initial estimate: 240×120–160 medium decisions ×1.3 s = 10.4–13.9
CPU-hours; 300 horizon days per load at 120–160 decisions ×(1.3+3.2) s = 45–60 CPU-hours.
At 24 cores, allow 1–1.5 hours for collection/training/selection, 3–4.5 hours for confirmation
including scenario16 and scheduling, and 0.5–1 hour for audit/bootstrap/reporting: approximately
**5–7 hours**, with substantial uncertainty from slow search states and process overhead.
The pipeline has a hard **8-hour wall-clock timeout**, including smoke and reporting, and
terminates its process group on timeout. A timeout is an incomplete run, no done marker;
never shrink days, omit seeds, or extend the budget to rescue it.

For this implementation request, run only the isolated tiny smoke and full pytest suite
with `PYTHONPATH=$PWD`. Prepare the full pipeline but **do not launch it**.
