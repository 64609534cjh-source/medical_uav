# DAgger implementation and execution notes

Protocol: [preregistration](../../specs/2026-10-28-dagger-preregistration.md), recorded before
data. Three iterations × 16 medium days × five expiry05 students, fixed initial-student
low/high replay, five epochs per aggregate. The horizon teacher uses visible information and
default v2 planning rewards. Deployment uses the existing single-forward-pass NetworkPolicy.

From the repository root:

```bash
export PYTHONPATH="$PWD"
PY=/home/caijihang/miniconda3/envs/medical-uav-rl/bin/python
"$PY" -m pytest -q
bash scripts/dagger_pipeline.sh smoke
```

Implementation checks on 2026-10-28:

- Tests were added and observed failing before the implementation, then passed.
- Full pytest suite: **334 passed in 135.35 seconds**. The six campaign tests were also
  rerun after adding the explicit iteration-zero/no-adaptation selection message.
- Actual expiry05 seed-11 checkpoint smoke: **3 seconds** reported by the pipeline;
  21 aggregate decisions, 7 medium horizon queries, 2 teacher/student disagreements.
  Collection plus one epoch took 1.09 seconds. Checkpoint reload and existing evaluator and
  selector compatibility passed. Artifacts: `/tmp/medical_uav_dagger_smoke.qDna3F/`.
- `bash -n scripts/dagger_pipeline.sh`, whitespace and 100-column code checks passed.
- No full campaign or real CONFIRM2 episodes were executed. Confirmation tests use synthetic
  count tables and mocked exports, with the original analysis bootstrap and target code.

The prepared full command is below; it was **not launched**:

```bash
bash scripts/dagger_pipeline.sh full
```

It runs smoke, five sequential students with 24 parallel teacher workers each, gated selection
on 100 selection days, then parallel CONFIRM2 exports (8 reference workers + 16 horizon workers).
It re-runs the horizon on the new days. Selection freezes paths and hashes before confirmation;
exports and reporting reject changed checkpoint hashes. Existing campaign artifacts are not
overwritten. Reporting calls the shared improvement analysis with explicit CONFIRM2 days and
the frozen-v2 reference set; statistical code is not duplicated.

Outputs: `runs/dagger_choice.json`, `runs/dagger_selection.csv`,
`runs/dagger_confirm2_{references,horizon}.csv`, `runs/dagger_report.{txt,json}`, then
`runs/dagger.done`. A completed report may contain failed targets; failed or timed-out
execution does not write the done marker. Training datasets/checkpoints/logs live under
`runs/dagger_set_s{11,22,33,44,55}` (logs beside the directories).

Estimated full wall-clock: **5–7 hours on 24 cores**, uncertain because planning states vary.
GNU timeout terminates the campaign process group at 28780 seconds and escalates after 10
seconds, leaving a margin within the eight-hour budget. The tiny smoke is a plumbing check;
its sparse early-day workload cannot validate the full-run throughput estimate.
