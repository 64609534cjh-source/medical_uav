# Post-locked-test improvement round — preregistration (2026-10-27)

Registered before any result from A, B or C, including pilots. Supersedes the destination-head
registration and combines all three experiments in this document. Existing locked-test findings
motivated this exploration: v2 PPO expired 6.2%/15.7% at medium/high; M1 control 2.7%/10.4%
with 3–7 percentage points less on-time; screen rolling horizon had approximately PPO on-time
and 3 points fewer expired. These are supplied historical facts, not new measurements.
This round cannot change earlier locked claims or entrants. No commit is permitted in this
worktree. No training, checkpoint selection or learned-policy episode is authorized during
implementation. Commands below are for a later authorized campaign.

## Factors and fixed protocol

| Arm / run prefix | Network | Reward | Table | Training maps | Matched frozen reference |
|---|---|---|---|---|---|
| A / `expiry03_set_s` | set | `v2_expiry03`: expiry −0.3 | yes | 0 | `m3_set_s*.pt` |
| A / `expiry05_set_s` | set | `v2_expiry05`: expiry −0.5 | yes | 0 | `m3_set_s*.pt` |
| B / `dest_set_s` | set_dest | v2: expiry −0.1 | yes | 0 | `m3_set_s*.pt` |
| C / `notable_set_s` | set | v2: expiry −0.1 | no | TRAIN_MAP_SEEDS 1000–1499 | `mm_set_s*.pt` |

One factor at a time relative to its matched reference. All rewards retain on-time +1,
late +0.1 (fixed), energy −0.001 per battery percent. C removes only the row-major 7×7
travel-time block: global 80→31, observation 452→403. The current-to-node travel times,
other global fields, pending/cargo fields, ordering, normalization, masks, physics and reward
remain identical. Networks are sized from the observation, with width and all other
hyperparameters unchanged. Checkpoints carry observation dimensions for future inference.
C is compared on **DEV_MAP_SEEDS 1–10 × all 100 selection days**, never maps 200–209.
Its secondary comparison is fresh days on map 0 against frozen mm and the common references.

All runs start from scratch: seeds 11,22,33,44,55; 6e6 decisions (2048-step updates round to
6000640); 16 environments × 128 steps, four PPO epochs, minibatch 512, gamma 1,
GAE lambda .95, clip .2, Adam epsilon 1e-5, LR 3e-4→5%, entropy .01→10%, reward /10.
Training days are RL_SEED_RANGE [1000,100000), loads uniformly drawn. Checkpoints every
5e5 decisions; curve evaluations use the first 20 selection days. For C these rotate through
SELECT_MAP_SEEDS 100–109, as in frozen multi-map training.

**B architecture.** New name `set_dest`; full-table SetNet and MLPNet architectures stay unchanged. Keep SetNet's width
64 encoders, four-head attention with null token, 128-wide shared context, PICK scorer and
value head. For each of the six facilities in DELIVER action order, concatenate context,
the destination's mean cargo embedding (64), mean visible pending embedding (64), cargo
count / 3, visible pending count / 12, current-to-facility travel time / 10 minutes as
already observed, and a learned facility embedding (8). Means exclude unoccupied slots
and are exactly zero for empty groups. One shared `267 -> 64 -> 1` MLP with ReLU after
the hidden layer scores all six DELIVER actions. A fixed `128 -> 64 -> 8` ReLU head scores
CHARGE, RETURN and WAIT. Apply the existing action mask. Use observation fields only:
pending occupancy 0 and destination 11:17, cargo occupancy 0 and destination 5:11,
global travel times 16:23 (facility nodes 0:6). No hidden orders or simulator state.
Parameter counts: `set_dest` 79803; existing `set` 62928.


## Stages and decisions (selection information only)

Stage 1 runs exactly four seed-11 pilots in parallel: A .3, A .5, B, C, each to the full
6e6 budget. Inspect crash/non-finite logs, approximate KL, clip fraction and learning curve.
For diagnostic context the usual KL is roughly 0–.05, clip fraction 0–.3; log medians,
95th percentiles and first/last-three selection-curve means. A flat/noisy curve or an isolated
KL excursion is not grounds for dropping or changing a run. Broken means crash, non-finite
model/loss/KL/clip (empty-episode train_on_time NaNs are permitted), or persistent numerical
failure. Report it and stop that arm; no hyperparameter repair or seed replacement.

Apply unchanged roadmap §4 selection: every saved checkpoint, all 100 SELECT_SEEDS
100040–100139, all loads; low on-time ≥97%; every type within 3 points of Slack at
medium/high; among eligible checkpoints greatest medium/high mean on-time, ties earlier.
A/B use map 0, C uses all SELECT_MAP_SEEDS 100–109 as frozen mm did. A missing eligible
checkpoint is a broken selection, reported explicitly; never substitute a final checkpoint.
For the stage-1 performance screen compare selected seed 11 to frozen m3 seed 11 on map 0;
C to frozen mm seed 11 on DEV_MAP_SEEDS 1–10, all SELECT_SEEDS. Drop only if broken or
medium/high mean on-time is **more than 2 points below** its matching frozen seed 11.
At exactly −2 points it survives. No adoption or gain claim is possible from one seed.
Record survivors and reasons in a selection-only manifest before stage 2.

Stage 2 runs seeds 22,33,44,55 of surviving arms, at most four runs at once, four threads
each. Retain pilot seed 11; all subsequent decisions require the five seeds. Failed selection
in stage 2 invalidates that arm; report failure, never omit a seed or tune it away.
Choose among v2 and surviving A variants using five-seed means on map 0 selection days
only: lowest medium/high expiry among variants whose medium/high on-time is no more than
0.5 points below frozen v2. Include v2 itself; expiry ties prefer v2, then .3, then .5.
If no new variant qualifies, keep v2. No fresh result can affect this choice.

B helps if its five-seed medium/high selection score minus frozen v2 is ≥0.5 points.
If and only if the chosen reward differs from v2 AND B helps, run **one** combined set,
`combined_set_s{seed}`: set_dest, chosen reward, table present, map 0, five seeds, same
6e6 budget, hyperparameters and §4 selection. This conditional set is registered now;
no reward×head grid, further combination with C, retries or fresh-driven choice.
Select the combined set as final candidate only if its five-seed selection score is ≥0.5
points above the chosen-reward set, with on-time no worse than v2−0.5 points and expiry
no higher than the chosen-reward set. Otherwise retain the chosen-reward set. When reward
stays v2, use B if B helps and its expiry is no higher than v2, otherwise frozen v2.
C is a separate ablation, never a fresh-driven replacement for this selected candidate.
Freeze the candidate identity, checkpoint paths and selection decisions before fresh access.

## Confirmation, reporting and explicit owner targets

Only FRESH_SEEDS 500000–500299, map 0, low/medium/high. Compare all completed five-seed
arms (including C secondary and combined if triggered), frozen m3, frozen mm for C,
slack, rollout, scenario16, and visible rolling horizon at medium/high. Existing baseline
jobs write runs/fresh_baselines.csv and runs/fresh_horizon.csv; implementation must not
read runs/fresh_*.csv. Never touch days 200000–200299, 300000–300099, 400000–400099,
locked maps 200–209, shipped checkpoints or frozen entrants. No inspection of fresh outcomes
until all selection decisions are fixed. A completed fresh comparison is a single exploratory
confirmation; failing a target is reported, **not tuned away**.

Report on-time, served and expired by load and medium/high mean, by training seed and per
order type. Served = 1−expired at episode completion; explicitly audit this and count identities.
Overall endpoints are equal-load means of daily fractions, then equal training-seed means.
Per-type endpoints pool counts over days (rather than averaging unstable rare-type daily rates).
Report absolute arm rates and differences in percentage points. 95% percentile bootstrap,
10000 draws, RNG 0: (1) paired whole days after averaging seeds and (2) two-way independent
resampling of five matched training-seed pairs and days, preserving arm/load pairing.
Deterministic references repeat the same day for every training seed; do not invent baseline
seeds. For C's development result average the ten fixed maps within each day before bootstrap;
per-map results are descriptive, inference is conditional on these ten development maps.
For per-type intervals recompute pooled numerator/denominator within each resample, retaining
all orders and map/day pairing. Report all pairwise reference contrasts and per-type intervals.
No silently missing/duplicate/non-finite rows, failed seeds, different demand counts or fabricated
per-type counts: stop and request complete data. Historical fresh baseline outputs may lack
served/expired per-type counts; obtain a supplemental baseline export using the same fixed
protocol before claiming complete per-type reporting. The visible-horizon implementation is
external to this worktree, so its exact re-export command cannot be specified here.

The three owner targets apply to the selection-frozen final candidate, on the equal medium/high
endpoint. Both paired-day and two-way lower bounds must meet each threshold:

1. **Beat every other method clearly:** on-time difference versus each of slack, rollout,
   scenario16, visible horizon and frozen v2 has lower bound >0. Report served superiority
   separately versus each reference (also lower bound >0), so an on-time win cannot hide expiry.
   Other completed improvement arms and frozen mm are also reported; the “every method”
   reading requires superiority to all distinct competing arms, excluding the candidate itself.
2. **Reach rolling-horizon on-time:** candidate minus visible horizon lower bound >−0.5 points
   (registered non-inferiority tolerance); label this tolerance explicitly, rather than claiming
   literal equality from a non-significant difference.
3. **Clearly lower expiry:** frozen v2 expiry minus candidate expiry point estimate ≥1 point
   AND lower bound >0 (equivalently candidate served minus frozen v2 served).

Print PASS/FAIL for each target and their conjunction, including failures and uncertainty.
No low-load or per-type secondary endpoint replaces a failed primary target. These are
pointwise intervals; the joint all-method success is an intersection of the specified tests.

## Prepared commands — future authorization only

All commands run from this worktree; never execute these campaigns as part of implementation.
The shell blocks below share these definitions:

```bash
set -euo pipefail
cd /home/caijihang/projects/medical_uav_dest
export PYTHONPATH="$PWD"
PY=/home/caijihang/miniconda3/envs/medical-uav-rl/bin/python
AN=docs/superpowers/experiments/2026-10-27-final/improvement_analysis.py
mkdir -p runs
train_one() {
  local arm=$1 seed=$2
  local net=set reward=v2
  local extra=()
  case "$arm" in
    expiry03) reward=v2_expiry03 ;;
    expiry05) reward=v2_expiry05 ;;
    dest) net=set_dest ;;
    notable) extra=(--maps train --no-travel-table) ;;
    combined) net=set_dest; reward=$CHOSEN_REWARD ;;
    *) return 2 ;;
  esac
  "$PY" scripts/train_ppo.py --net "$net" --reward "$reward" --seed "$seed" \
    --decisions 6e6 --threads 4 --eval-every 5e5 --eval-days 20 "${extra[@]}" \
    --out "runs/${arm}_set_s$seed" > "runs/${arm}_set_s$seed.log" 2>&1
}
wait_batch() {
  local status=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}
select_arm() {
  local arm=$1; shift
  local maps=()
  if [[ "$arm" == notable ]]; then maps=(--maps select); fi
  "$PY" scripts/select_checkpoints.py "$@" --days 100 --processes 4 \
    "${maps[@]}" --out "runs/selection_${arm}.json"
}
```

### Stage 1 (four pilots; inspect all four even if one fails)

```bash
pids=()
for arm in expiry03 expiry05 dest notable; do
  train_one "$arm" 11 & pids+=("$!")
done
wait_batch || echo 'Pilot failure: record broken arm; do not restart'
for arm in expiry03 expiry05 dest notable; do
  select_arm "$arm" "runs/${arm}_set_s11" || echo "Selection failed: $arm; record failure"
done
```

Export selected checkpoints and frozen references on selection days (A/B map 0;
C dev maps) using the analysis script's `--export-selection` mode, which prepares/runs
only those selection evaluations when separately authorized:

```bash
for arm in expiry03 expiry05 dest notable; do
  "$PY" "$AN" --export-selection --arm "$arm" --selection "runs/selection_${arm}.json" \
    || echo "Export failed: $arm; inspect selection and logs before pilot decision"
done
"$PY" "$AN" --pilot --selection-dir runs --out runs/improvement_pilot.json
```

The pilot command audits logs/curves and selected-checkpoint direction; inspect its diagnostics,
record broken-arm evidence, and freeze `survivors` in runs/improvement_pilot.json. No gain
claim is made. Absent eligible selections or incomplete output is an explicit failure.

### Stage 2 (one four-seed batch per surviving arm)

```bash
mapfile -t SURVIVORS < <("$PY" -c 'import json; print("\n".join(json.load(open("runs/improvement_pilot.json"))["survivors"]))')
for arm in "${SURVIVORS[@]}"; do
  pids=()
  for s in 22 33 44 55; do train_one "$arm" "$s" & pids+=("$!"); done
  wait_batch
  select_arm "$arm" runs/"${arm}"_set_s{11,22,33,44,55}
  "$PY" "$AN" --export-selection --arm "$arm" --selection "runs/selection_${arm}.json"
done
# Reference-only export also covers the case that all single-map pilots were dropped.
"$PY" "$AN" --export-selection --arm frozen
"$PY" "$AN" --choose --pilot-manifest runs/improvement_pilot.json \
  --selection-dir runs --out runs/improvement_choice.json
```

If improvement_choice.json has `combine: true`, and only then:

```bash
CHOSEN_REWARD=$("$PY" -c 'import json; print(json.load(open("runs/improvement_choice.json"))["reward"])')
pids=()
for s in 11 22 33 44; do train_one combined "$s" & pids+=("$!"); done
wait_batch
train_one combined 55
select_arm combined runs/combined_set_s{11,22,33,44,55}
"$PY" "$AN" --export-selection --arm combined --selection runs/selection_combined.json
"$PY" "$AN" --choose --pilot-manifest runs/improvement_pilot.json \
  --selection-dir runs --include-combined --out runs/improvement_choice.json
```

Freeze this manifest before confirmation. Then (only after separate campaign authorization):

```bash
"$PY" "$AN" --export-fresh --choice runs/improvement_choice.json --selection-dir runs
"$PY" "$AN" --confirm --choice runs/improvement_choice.json --selection-dir runs \
  --csv runs/improvement_fresh.csv runs/fresh_baselines.csv runs/fresh_horizon.csv \
  --out runs/improvement_report.json
```

Fresh exporter evaluates selected learned checkpoints and frozen m3/mm only, with no baseline
re-runs. Existing baseline/horizon CSVs are read only by the future confirm command. Missing
per-type counts require supplemental exports, never reconstruction from aggregate rates.

After C's five selections and development export are complete, report its primary ablation:

```bash
"$PY" "$AN" --development --selection-dir runs --out runs/improvement_C_development.json
```

If the external horizon CSV uses a method label other than `horizon`, pass its exact label
with `--horizon-label LABEL` to `--confirm`. This names the registered visible-horizon method;
it does not select among optimizer versions. Confirm's input files must together contain
exactly the declared entrant grid, with no overlapping copies of the same entrant/day row.
