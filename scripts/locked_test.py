"""The final test: freeze the registered entrants, then run every day of them once.

Registered in docs/superpowers/specs/2026-10-29-final-test-preregistration.md, which supersedes
the 10/27 locked test (its manifest and results stay in git history). `freeze` copies the
selected checkpoints to checkpoints/entrants/ and the shipped agent to checkpoints/ppo_set.pt,
and writes the manifest (entrants, checkpoint hashes, days, versions). `run` checks the hashes
and evaluates every (block, entrant, map, load, day), one JSON file per day written atomically
and bound to the manifest, the code and the days, so an interrupted run resumes where it stopped
and a file of another campaign is refused. `--days select` rehearses on the selection days and
maps; only `--days test` opens the test days and maps.

Usage:
    python scripts/locked_test.py freeze
    python scripts/locked_test.py run --days select --limit 2 --out runs/final_rehearsal
    python scripts/locked_test.py run --days test --out runs/final_test
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
from medical_uav.splits import (FINAL_MAP_DAY0, FINAL_MAP_DAYS, FINAL_TEST_MAP_SEEDS,
                                FINAL_TEST_SEEDS, SELECT_MAP_SEEDS, SELECT_SEEDS)

import evaluate  # scripts/evaluate.py: the rule-based policies

REGISTRATION = "docs/superpowers/specs/2026-10-29-final-test-preregistration.md"
MANIFEST = Path("checkpoints/entrants/manifest.json")
ENTRANTS = Path("checkpoints/entrants")
SHIPPED = Path("checkpoints/ppo_set.pt")
NO_FULL = "#no100"  # a checkpoint run with CHARGE(*, 100) hidden
RULES = ["random", "edf", "fcfs", "nearest", "slack", "insertion", "rollout", "scenario16",
         "horizon", "slack100", "prepeak"]
SLOW = ("scenario16", "horizon")  # run first
SKIP = [["horizon", "low"]]  # registered: the rolling horizon runs medium and high only
SEEDS = (11, 22, 33, 44, 55)
FINAL = [f"f12_mm_e05_s{s}" for s in SEEDS]  # the final agent; s11 is shipped
PATH = [f"f12_{v}_s{s}" for v in ("mm_v2", "sm_e05", "sm_v2") for s in SEEDS]
CONTROLS = [f"{v}_s{s}" for v in ("f12_sm_v2_mlp", "f24_sm_v2_mlp", "f12_mm_e05_mlp",
                                   "f24_mm_e05_mlp", "f12_sm_e05_bc", "f12_sm_m1",
                                   "f12_sm_ontime") for s in SEEDS[:3]]
IMITATION = ["bc_set", "bc_mlp", "bc_rollout_set", "bc_rollout_mlp"]
SELECTIONS = ["runs/f12_select_mm.json", "runs/f12_select_sm.json", "runs/f12_select_ctrl.json",
              "runs/f12_select_mmmlp.json"]
LOADS = ("low", "medium", "high")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks(days: str) -> dict:
    """{block: [(map, demand days)]}: map 0, and the unseen maps, each with its own test days
    (the rehearsal shares the first selection days across the selection maps)."""
    if days == "test":
        return {"map0": [(0, FINAL_TEST_SEEDS)],
                "maps": [(m, range(FINAL_MAP_DAY0 + FINAL_MAP_DAYS * i,
                                   FINAL_MAP_DAY0 + FINAL_MAP_DAYS * (i + 1)))
                         for i, m in enumerate(FINAL_TEST_MAP_SEEDS)]}
    return {"map0": [(0, SELECT_SEEDS)],
            "maps": [(m, SELECT_SEEDS[:FINAL_MAP_DAYS]) for m in SELECT_MAP_SEEDS]}


def describe(cells: dict) -> dict:
    """The blocks as [map, first day, last day] lists, as stored in the manifest."""
    return {name: [[m, days[0], days[-1]] for m, days in maps] for name, maps in cells.items()}


def freeze() -> None:
    """Copy the selected checkpoints to checkpoints/entrants/ (the 10/27 entrants are removed;
    git keeps them), ship the final agent's seed 11 and write the manifest."""
    selected = {}
    for f in SELECTIONS:
        selected.update({Path(k).name: v["selected"]
                         for k, v in json.loads(Path(f).read_text())["runs"].items()})
    for old in ENTRANTS.glob("*.pt"):
        old.unlink()
    entrants = {r: {"policy": r} for r in RULES}
    for run in FINAL + PATH + CONTROLS:
        path = ENTRANTS / f"{run}.pt"
        shutil.copyfile(selected[run], path)
        entrants[run] = {"policy": str(path), "sha256": sha256(path), "source": selected[run]}
    for name in IMITATION:
        path = f"checkpoints/{name}.pt"
        entrants[name] = {"policy": path, "sha256": sha256(path)}
    for run in FINAL:
        entrants[f"{run}/no100"] = {**entrants[run], "policy": entrants[run]["policy"] + NO_FULL}
    shutil.copyfile(ENTRANTS / f"{FINAL[0]}.pt", SHIPPED)
    manifest = {
        "registered": [REGISTRATION],
        "shipped": {"entrant": FINAL[0], "file": str(SHIPPED), "sha256": sha256(SHIPPED)},
        "entrants": entrants,
        "skip": SKIP,
        "days": {k: describe(blocks(k)) for k in ("select", "test")},
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
    cells = blocks(days)
    if describe(cells) != manifest["days"][days]:
        raise SystemExit("the days differ from the manifest")
    skip = {tuple(x) for x in manifest["skip"]}
    todo = []
    for name, maps in cells.items():
        for label, entrant in manifest["entrants"].items():
            folder = out / name / label.replace("/", "__")
            for m, seeds in maps:
                for load in LOADS:
                    if (label, load) in skip:
                        continue
                    for day in list(seeds)[:limit]:
                        todo.append((name, label, entrant["policy"], m, load, day,
                                     folder / f"{m}_{load}_{day}.json"))
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
        raise SystemExit("the final test runs every day")
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
        todo.sort(key=lambda j: j[0][1] not in SLOW)  # the slow planners' days first
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
    parser.add_argument("--days", choices=["select", "test"], default="select")
    parser.add_argument("--limit", type=int, default=None, help="at most LIMIT days per map")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--out", default="runs/final_rehearsal")
    args = parser.parse_args()
    freeze() if args.command == "freeze" else run(args)


if __name__ == "__main__":
    main()
