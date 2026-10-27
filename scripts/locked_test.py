"""The locked test: freeze the registered entrants, then run every day of them once.

Registered in docs/superpowers/specs/2026-10-27-final-phase-preregistration.md (item 3) and
2026-10-27-multimap-preregistration.md. `freeze` copies the selected checkpoints to
checkpoints/entrants/ and writes the manifest (entrants, checkpoint hashes, blocks of days,
versions). `run` checks the hashes and evaluates every (entrant, map, load, day) of the
manifest's blocks, one JSON file per day written atomically and bound to the manifest, the code
and the days, so an interrupted run resumes where it stopped and a file of another campaign is
refused. `--days select` rehearses on the selection days and maps; only
`--days test` opens the locked test days and maps.

Usage:
    python scripts/locked_test.py freeze
    python scripts/locked_test.py run --days select --limit 5 --out runs/locked_rehearsal
    python scripts/locked_test.py run --days test --out runs/locked
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.energy import PrePeakTopUp, slack100_policy, without_full_charge
from medical_uav.baselines.scenario import MASTER_SEED
from medical_uav.env import EnvConfig
from medical_uav.evaluation import heuristic, run_day, run_metadata
from medical_uav.splits import SELECT_MAP_SEEDS, SELECT_SEEDS, TEST_MAP_SEEDS, TEST_SEEDS

import evaluate  # scripts/evaluate.py: the rule-based policies

MANIFEST = Path("checkpoints/entrants/manifest.json")
ENTRANTS = Path("checkpoints/entrants")
NO_FULL = "#no100"  # a checkpoint run with CHARGE(*, 100) hidden
RULES = ["random", "edf", "fcfs", "nearest", "slack", "rollout", "scenario16", "slack100",
         "prepeak"]
SEEDS = (11, 22, 33, 44, 55)
PRIMARY = [f"m3_set_s{s}" for s in SEEDS]  # single-map PPO; s11 is the shipped model
MULTIMAP = [f"mm_set_s{s}" for s in SEEDS]
CONTROLS = [f"m3_{v}_s{s}" for v in ("mlp", "set_m1", "set_ontime", "set_bc") for s in SEEDS[:3]]
DAYS = {"select": (SELECT_SEEDS, SELECT_MAP_SEEDS), "test": (TEST_SEEDS, TEST_MAP_SEEDS)}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze() -> None:
    """Copy the selected checkpoints next to the submitted ones and write the manifest."""
    selected = {Path(k).name: v for k, v in
                json.loads(Path("outputs/m3/selected.json").read_text()).items()}
    selected.update({Path(k).name: v["selected"] for k, v in
                     json.loads(Path("runs/selection_mm.json").read_text())["runs"].items()})
    ENTRANTS.mkdir(parents=True, exist_ok=True)
    entrants = {r: {"policy": r} for r in RULES}
    for run in PRIMARY + MULTIMAP + CONTROLS:
        path = ENTRANTS / f"{run}.pt"
        shutil.copyfile(selected[run], path)
        entrants[run] = {"policy": str(path), "sha256": sha256(path), "source": selected[run]}
    for name in ("bc_set", "bc_mlp"):
        path = f"checkpoints/{name}.pt"
        entrants[name] = {"policy": path, "sha256": sha256(path)}
    for run in PRIMARY:
        entrants[f"{run}/no100"] = {**entrants[run], "policy": entrants[run]["policy"] + NO_FULL}
    manifest = {
        "registered": ["docs/superpowers/specs/2026-10-27-final-phase-preregistration.md",
                       "docs/superpowers/specs/2026-10-27-multimap-preregistration.md"],
        "shipped": {"entrant": "m3_set_s11", "file": "checkpoints/ppo_set.pt",
                    "sha256": sha256("checkpoints/ppo_set.pt")},
        "entrants": entrants,
        # map 0: every entrant, every day, every load; new maps: both PPO arms and two rules,
        # the first 30 days on each map, medium and high
        "blocks": [{"name": "map0", "entrants": list(entrants), "maps": "map 0",
                    "loads": ["low", "medium", "high"], "days": "all"},
                   {"name": "maps", "entrants": PRIMARY + MULTIMAP + ["slack", "rollout"],
                    "maps": "all", "loads": ["medium", "high"], "days": 30}],
        "days": {k: [days[0], days[-1], maps[0], maps[-1]] for k, (days, maps) in DAYS.items()},
        "rng": {"scenario16": f"master seed {MASTER_SEED}, streams [seed, decision, scenario]",
                "random": "seeded by each day's demand seed", "bootstrap": "seed 0, 10 000"},
        "threads": "one torch thread per worker",
        "metadata": run_metadata({"numpy": np.__version__}),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(entrants)} entrants; wrote {MANIFEST}")


def make_policy(spec: str):
    """policy(env, obs, mask) from a manifest entry, and the pre-peak rule itself (its top-ups
    are counted)."""
    if spec == "prepeak":
        rule = PrePeakTopUp()
        return heuristic(rule), rule
    if spec == "slack100":
        return heuristic(slack100_policy), None
    if spec in RULES:
        return next(iter(evaluate.make_policies([spec], []).values())), None
    net = NetworkPolicy(load_checkpoint(spec.removesuffix(NO_FULL))[0])
    return (without_full_charge(net) if spec.endswith(NO_FULL) else net), None


def fingerprint(days: str, limit: int | None) -> dict:
    """What every result of a campaign is bound to: the manifest, the code that produces the
    results (committed, hashed by content) and the days."""
    files = sorted(Path("medical_uav").rglob("*.py")) + [Path("scripts/locked_test.py"),
                                                          Path("scripts/evaluate.py")]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *map(str, files), str(MANIFEST)],
                           capture_output=True, text=True).stdout
    if dirty:
        raise SystemExit(f"commit these first:\n{dirty}")
    code = hashlib.sha256(b"".join(str(f).encode() + f.read_bytes() for f in files)).hexdigest()
    return {"manifest": sha256(MANIFEST), "code": code, "days": days, "limit": limit}


def expected(manifest: dict, days: str, limit: int | None, out: Path) -> list:
    """Every (block, entrant, policy, map, load, day, file) of the campaign."""
    seeds, maps = DAYS[days]
    if [seeds[0], seeds[-1], maps[0], maps[-1]] != manifest["days"][days]:
        raise SystemExit("the day ranges differ from the manifest")
    todo = []
    for b in manifest["blocks"]:
        block_days = list(seeds)[:b["days"] if b["days"] != "all" else None][:limit]
        block_maps = [0] if b["maps"] == "map 0" else list(maps)
        for label in b["entrants"]:
            folder = out / b["name"] / label.replace("/", "__")
            for m in block_maps:
                for load in b["loads"]:
                    for day in block_days:
                        todo.append((b["name"], label, manifest["entrants"][label]["policy"],
                                     m, load, day, folder / f"{m}_{load}_{day}.json"))
    return todo


def check_saved(job, campaign: dict) -> None:
    """A saved day must belong to this campaign and to this job."""
    row = json.loads(job[6].read_text())
    if row["campaign"] != campaign or (row["block"], row["method"], row["map"], row["load"],
                                       row["seed"]) != (job[0], job[1], job[3], job[4], job[5]):
        raise SystemExit(f"{job[6]} belongs to another campaign or day")


def run_job(job) -> None:
    """One day of one entrant; the row is written to a temporary file, then renamed."""
    torch.set_num_threads(1)
    (block, label, spec, map_seed, load, day, path), campaign = job
    policy, rule = make_policy(spec)
    row = run_day(policy, load, day, EnvConfig(map_seed=map_seed), details=True)
    row = {"block": block, "method": label, "map": map_seed, **row,
           "top_ups": len(rule.used) if rule else None, "campaign": campaign}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, path)


def run(args) -> None:
    if args.days == "test" and args.limit is not None:
        raise SystemExit("the locked test runs every day")
    manifest = json.loads(MANIFEST.read_text())
    for label, e in manifest["entrants"].items():
        if "sha256" in e and sha256(e["policy"].removesuffix(NO_FULL)) != e["sha256"]:
            raise SystemExit(f"{label}: the checkpoint differs from the manifest")
    campaign, out = fingerprint(args.days, args.limit), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".lock"  # one writer at a time; left behind by a crash, remove it by hand
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    try:
        todo = []
        for job in expected(manifest, args.days, args.limit, out):
            job[6].parent.mkdir(parents=True, exist_ok=True)
            if job[6].exists():
                check_saved(job, campaign)
            else:
                todo.append((job, campaign))
        todo.sort(key=lambda j: j[0][1] != "scenario16")  # the slow planner's days first
        print(f"{len(todo)} days to run", flush=True)
        with Pool(args.processes) as pool:
            for k, _ in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
                if k % 500 == 0 or k == len(todo):
                    print(f"{k}/{len(todo)}", flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--days", choices=list(DAYS), default="select")
    parser.add_argument("--limit", type=int, default=None, help="at most LIMIT days per block")
    parser.add_argument("--processes", type=int, default=8, help="registered maximum: 8")
    parser.add_argument("--out", default="runs/locked_rehearsal")
    args = parser.parse_args()
    freeze() if args.command == "freeze" else run(args)


if __name__ == "__main__":
    main()
