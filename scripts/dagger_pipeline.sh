#!/usr/bin/env bash
# Default is the isolated smoke. Full training requires the explicit `full` argument.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PYTHON:-/home/caijihang/miniconda3/envs/medical-uav-rl/bin/python}

smoke() {
  local started=$SECONDS smoke_dir init
  smoke_dir=$(mktemp -d /tmp/medical_uav_dagger_smoke.XXXXXX)
  init=$("$PY" -c 'import json; print(json.load(open("runs/improvement_choice.json"))[
"paths"]["expiry05"][0])')
  "$PY" scripts/train_dagger.py --init "$init" --seed 11 --iterations 1 \
    --days-per-iteration 1 --processes 2 --smoke --out "$smoke_dir/student"
  "$PY" scripts/dagger_campaign.py smoke --directory "$smoke_dir/student" \
    --out "$smoke_dir/compatibility.json"
  echo "Smoke PASS: $((SECONDS - started)) s; artifacts: $smoke_dir"
}

case "${1:-smoke}" in
  smoke) smoke; exit 0 ;;
  full)
    # GNU timeout signals the complete campaign process group, including worker pools.
    exec timeout --signal=TERM --kill-after=10s 28780s \
      env DAGGER_BUDGETED=1 bash "$0" _full
    ;;
  _full) [[ ${DAGGER_BUDGETED:-} == 1 ]] || exit 2 ;;
  *) echo "Usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

for artifact in dagger.done dagger_choice.json dagger_report.json dagger_report.txt \
  dagger_confirm2_references.csv dagger_confirm2_horizon.csv dagger_selection.csv; do
  if [[ -e runs/$artifact ]]; then
    echo "Refusing to overwrite runs/$artifact" >&2
    exit 1
  fi
done
for seed in 11 22 33 44 55; do
  if [[ -e runs/dagger_set_s$seed ]]; then
    echo "Refusing to overwrite runs/dagger_set_s$seed" >&2
    exit 1
  fi
done

smoke
mapfile -t initial < <("$PY" -c 'import json; print("\n".join(
json.load(open("runs/improvement_choice.json"))["paths"]["expiry05"]))')
seeds=(11 22 33 44 55)
[[ ${#initial[@]} == 5 ]]
for i in "${!seeds[@]}"; do
  seed=${seeds[$i]}
  "$PY" scripts/train_dagger.py --init "${initial[$i]}" --seed "$seed" \
    --iterations 3 --days-per-iteration 16 --processes 24 --out "runs/dagger_set_s$seed" \
    > "runs/dagger_set_s$seed.log" 2>&1
done
"$PY" scripts/dagger_campaign.py select --processes 24 --out runs/dagger_choice.json

# Only after the complete choice is frozen: 8 + 16 single-thread workers <= 24 cores.
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
"$PY" scripts/dagger_campaign.py export --group references --processes 8 \
  --out runs/dagger_confirm2_references.csv > runs/dagger_confirm2_references.log 2>&1 &
pids+=("$!")
"$PY" scripts/dagger_campaign.py export --group horizon --processes 16 \
  --out runs/dagger_confirm2_horizon.csv > runs/dagger_confirm2_horizon.log 2>&1 &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status == 0)) || exit 1
pids=()
"$PY" scripts/dagger_campaign.py confirm \
  --csv runs/dagger_confirm2_references.csv runs/dagger_confirm2_horizon.csv \
  --out runs/dagger_report.json > runs/dagger_report.txt
"$PY" -c 'import json; from pathlib import Path
r=json.load(open("runs/dagger_report.json"))
assert r["n_boot"] == 10000 and len(r["confirmation_days"]) == 300
Path("runs/dagger.done").write_text("Complete audited confirmation; see dagger_report.txt/json\n")'
echo "DAgger campaign complete: runs/dagger_report.txt/json"
