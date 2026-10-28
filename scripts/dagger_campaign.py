"""Selection, new-day exports and confirmation for the preregistered DAgger campaign."""
import argparse
import csv
import hashlib
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from medical_uav.agents.dagger import day_schedule
from medical_uav.evaluation import checkpoint_labels, run_metadata, select_checkpoint
from medical_uav.splits import CONFIRM2_SEEDS, SELECT_SEEDS, TRAINING_SEEDS
from scripts.evaluate import run_job
from scripts.select_checkpoints import evaluate_job, summarize

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "docs/superpowers/experiments/2026-10-27-final/improvement_analysis.py"
SPEC = importlib.util.spec_from_file_location("improvement_analysis", ANALYSIS)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
LOADS = ("low", "medium", "high")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_hashes(choice):
    for path, digest in choice["checkpoint_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Frozen checkpoint changed: {path}")


def choose_checkpoint(results, baseline):
    """Medium on-time first, with matched expiry05 high on-time and mean expiry gates."""
    rows = []
    for path, rates in sorted(results.items()):
        if not all(np.isfinite(v) for r in (baseline, rates) for load in LOADS
                   for v in r[load].values()):
            raise ValueError("Selection rates must be finite")
        expiry = np.mean([rates[load]["expiry_rate"] for load in LOADS[1:]])
        base_expiry = np.mean([baseline[load]["expiry_rate"] for load in LOADS[1:]])
        high = rates["high"]["on_time_rate"] >= baseline["high"]["on_time_rate"] - .005 - 1e-12
        expired = expiry <= base_expiry + .003 + 1e-12
        rows.append({"path": path, "eligible": bool(high and expired),
                     "high_gate": bool(high), "expiry_gate": bool(expired),
                     "medium_on_time": rates["medium"]["on_time_rate"],
                     "expiry": float(expiry),
                     "served": float(np.mean([rates[l]["served_rate"] for l in LOADS[1:]]))})
    eligible = [r for r in rows if r["eligible"]]
    best = max(eligible, key=lambda r: (r["medium_on_time"], -r["expiry"], r["served"]),
               default=None)
    if best is None:
        raise ValueError("No eligible checkpoint, including iteration zero")
    return best["path"], rows


def select(source, directory, processes):
    """Evaluate all registered checkpoints, then freeze paths and hashes before new days."""
    if source.get("candidate") != "expiry05" or not source.get("ready_for_fresh"):
        raise ValueError("Expected the selection-frozen expiry05 initialization choice")
    initial = source["paths"]["expiry05"]
    analysis.entrants({"runs": {f"runs/expiry05_set_s{s}": {"selected": p}
                                for s, p in zip(TRAINING_SEEDS, initial)}}, "expiry05")
    if len(initial) != 5 or source["paths"]["frozen"] != analysis.frozen_paths():
        raise ValueError("Initial/frozen references do not match the five registered seeds")
    runs = [Path(directory) / f"dagger_set_s{s}" for s in TRAINING_SEEDS]
    candidates = {}
    for seed, run, init in zip(TRAINING_SEEDS, runs, initial):
        meta = json.loads((run / "run.json").read_text())
        medium, replay = day_schedule(seed, 3, 16)
        if (meta["smoke"] or meta["seed"] != seed or meta["iterations"] != 3
                or meta["days_per_iteration"] != 16 or meta["medium_days"] != medium
                or meta["replay_days"] != replay or meta["init"] != init
                or meta["init_sha256"] != file_hash(init)):
            raise ValueError(f"Unregistered training run: {run}")
        paths = [str(run / f"ckpt_{i:08d}.pt") for i in range(4)]
        if sorted(str(p) for p in run.glob("ckpt_*.pt")) != paths:
            raise ValueError(f"Expected exactly four iteration checkpoints: {run}")
        candidates[str(run)] = paths
    paths = initial + [p for group in candidates.values() for p in group]
    jobs = [(path, load, 0, list(SELECT_SEEDS)) for path in paths for load in LOADS]
    with Pool(processes) as pool:
        done = pool.map(evaluate_job, jobs, chunksize=1)
    rows = [{"method": name, **row} for name, _, day_rows in done for row in day_rows]
    raw = Path(directory) / "dagger_selection.csv"
    write_rows(raw, rows)
    analysis.read_rows([raw], paths, SELECT_SEEDS)
    results = {}
    for name, load, day_rows in done:
        results.setdefault(name, {})[load] = summarize(day_rows)
    records, selected = {}, []
    for (run, group), init in zip(candidates.items(), initial):
        chosen, audit = choose_checkpoint({p: results[p] for p in group}, results[init])
        adapted = chosen != group[0]
        records[run] = {"selected": chosen, "initial": init, "adapted": adapted,
                        "checkpoints": audit}
        status = "adaptation selected" if adapted else "no adaptation selected"
        print(f"{run}: {chosen}; {status}")
        selected.append(chosen)
    references = {"dagger": selected, "expiry05": initial, "frozen": source["paths"]["frozen"]}
    frozen = {p: file_hash(p) for group in references.values() for p in group}
    return {"selection_only": True, "ready_for_fresh": True, "candidate": "dagger",
            "completed": ["dagger", "expiry05"], "paths": references,
            "runs": records, "all_results": results, "checkpoint_sha256": frozen,
            "confirmation_days": [CONFIRM2_SEEDS.start, CONFIRM2_SEEDS.stop],
            "metadata": run_metadata({"selection_days": list(SELECT_SEEDS)})}


def validate_choice(choice):
    if (choice.get("candidate") != "dagger" or choice.get("completed") != ["dagger", "expiry05"]
            or choice.get("confirmation_days") != [600000, 600300]):
        raise ValueError("Expected a frozen DAgger choice on CONFIRM2 days")
    return analysis.validate_choice(choice, references=("frozen",))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_rows(choice, group, processes, out):
    validate_choice(choice)
    if group == "horizon":
        named, loads = [("horizon", None, "horizon")], LOADS[1:]
    elif group == "references":
        paths = [p for group in choice["paths"].values() for p in group]
        named = [(m, None, m) for m in ("slack", "rollout", "scenario16")]
        named += [(None, p, label) for p, label in zip(paths, checkpoint_labels(paths))]
        loads = LOADS
    else:
        raise ValueError("Unknown export group")
    jobs = [(m, p, label, load, [day]) for load in loads for m, p, label in named
            for day in CONFIRM2_SEEDS]
    if processes > 1:
        with Pool(processes) as pool:
            rows = [r for part in pool.map(run_job, jobs, chunksize=1) for r in part]
    else:
        rows = [r for job in jobs for r in run_job(job)]
    write_rows(out, rows)


def confirm(choice, paths, n_boot=10000):
    validate_choice(choice)
    result = analysis.confirm(choice, paths, days=CONFIRM2_SEEDS, references=("frozen",),
                              n_boot=n_boot)
    result["confirmation_days"] = list(CONFIRM2_SEEDS)
    result["choice"] = choice
    result["csv_sha256"] = {str(p): file_hash(p) for p in paths}
    expiry = result["contrasts"]["expiry05"]["medium+high"]["all/expired"]
    high = result["contrasts"]["expiry05"]["high"]["all/on_time"]
    result["retention"] = {
        "expiry_no_worse_point": expiry["two_way"][0] <= 0,
        "expiry_no_worse_both_upper_bounds": all(ci[2] <= 0 for ci in expiry.values()),
        "expiry_within_selection_tolerance": expiry["two_way"][0] <= .003 + 1e-12,
        "high_on_time_within_selection_tolerance": high["two_way"][0] >= -.005 - 1e-12,
        "expiry_tolerance_both_upper_bounds": all(ci[2] <= .003 for ci in expiry.values()),
        "high_tolerance_both_lower_bounds": all(ci[1] >= -.005 for ci in high.values())}
    result["literal_horizon_on_time_point"] = {
        load: result["contrasts"]["horizon"][load]["all/on_time"]["two_way"][0] >= 0
        for load in ("medium", "high", "medium+high")}
    print("expiry05 retention:", json.dumps(result["retention"]))
    print("Literal on-time >= horizon (point):",
          json.dumps(result["literal_horizon_on_time_point"]))
    return result


def smoke(directory):
    """Exercise the existing evaluator/selector readers on reserved smoke days only."""
    paths = sorted(str(p) for p in Path(directory).glob("ckpt_*.pt"))
    if len(paths) != 2:
        raise ValueError("Smoke expects the initial and one adapted checkpoint")
    results = {}
    for path in ["slack"] + paths:
        for load in LOADS:
            name, _, rows = evaluate_job((path, load, 0, [709905]))
            results.setdefault(name, {})[load] = summarize(rows)
    selected, audit = select_checkpoint({p: results[p] for p in paths}, results["slack"])
    exported = run_job((None, paths[-1], "smoke_student", "medium", [709905]))
    if len(exported) != 1 or exported[0]["method"] != "smoke_student":
        raise ValueError("Existing evaluator failed to export the DAgger checkpoint")
    print("Smoke checkpoint reload, existing selector and evaluator: PASS")
    return {"scope": "reserved smoke days only; no performance selection",
            "compatible": True, "selected": selected, "checkpoints": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "select", "export", "confirm"))
    parser.add_argument("--source", default="runs/improvement_choice.json")
    parser.add_argument("--directory", default="runs")
    parser.add_argument("--choice", default="runs/dagger_choice.json")
    parser.add_argument("--processes", type=int, default=24)
    parser.add_argument("--group", choices=("references", "horizon"))
    parser.add_argument("--csv", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 1 <= args.processes <= 24:
        parser.error("--processes must be in 1..24")
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite a campaign artifact: {out}")
    if args.mode == "smoke":
        result = smoke(args.directory)
    elif args.mode == "select":
        result = select(json.loads(Path(args.source).read_text()), args.directory, args.processes)
    else:
        choice = json.loads(Path(args.choice).read_text())
        audit_hashes(choice)
        if args.mode == "export":
            export_rows(choice, args.group, args.processes, out)
            return
        if not args.csv:
            parser.error("confirm requires --csv")
        result = confirm(choice, args.csv)
    out.write_text(json.dumps(result, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
