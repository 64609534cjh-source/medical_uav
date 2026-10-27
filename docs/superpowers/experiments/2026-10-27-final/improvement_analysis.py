"""Selection-only decisions and registered fresh confirmation; export modes run policies.

Default analysis modes only read completed exports, never checkpoints or policy episodes.
See 2026-10-27-improvement-round-preregistration.md for future campaign commands.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from medical_uav.env.config import ENV_VERSION, ORDER_TYPES
from medical_uav.evaluation import checkpoint_labels
from medical_uav.splits import DEV_MAP_SEEDS, FRESH_SEEDS, SELECT_SEEDS, TRAINING_SEEDS

LOADS = ("low", "medium", "high")
ARMS = ("expiry03", "expiry05", "dest", "notable")
REWARD_ARMS = {"v2": "frozen", "v2_expiry03": "expiry03", "v2_expiry05": "expiry05"}
N_BOOT = 10_000


def frozen_paths(multimap=False, seeds=TRAINING_SEEDS):
    return [f"checkpoints/entrants/{'mm' if multimap else 'm3'}_set_s{s}.pt" for s in seeds]


def entrants(selection, arm, seeds=TRAINING_SEEDS):
    """Reject failed selection, missing/extra seeds, and paths from other run directories."""
    by_name = {Path(run).name: record for run, record in selection["runs"].items()}
    expected = [f"{arm}_set_s{s}" for s in seeds]
    if len(selection["runs"]) != len(expected) or set(by_name) != set(expected):
        raise ValueError(f"Selection must contain exactly {expected}")
    selected = [by_name[name]["selected"] for name in expected]
    if any(not p for p in selected):
        raise ValueError(f"Selection failure in {arm}")
    if any(Path(p).parent.name != n for p, n in zip(selected, expected)):
        raise ValueError("Selected checkpoint does not belong to its registered run")
    if len(set(selected)) != len(selected):
        raise ValueError("Repeated selected checkpoint")
    return selected


def load_json(path):
    return json.loads(Path(path).read_text())


def selection_path(directory, arm):
    return Path(directory) / f"selection_{arm}.json"


def export_path(directory, arm):
    return Path(directory) / f"improvement_selection_{arm}.csv"


def read_rows(paths, methods, days, maps=(0,), horizon=None):
    """Require the exact registered grid and coherent overall and per-type outcome counts.

    Horizon is permitted only at medium/high; all other methods require every load.
    Multiple input files may have different columns but must not overlap in row identity.
    """
    expected = {(m, load, map_seed, day) for m in methods
                for load in (LOADS[1:] if m == horizon else LOADS)
                for map_seed in maps for day in days}
    rows = {}
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["method"], row["load"], int(row.get("map", 0)), int(row["seed"])
                if key not in expected or key in rows:
                    raise ValueError(f"Unexpected or duplicate method/load/map/day: {key}")
                if row["env_version"] != ENV_VERSION:
                    raise ValueError("Expected environment v2")
                rates = [float(row[k]) for k in ("on_time_rate", "served_rate", "expiry_rate")]
                if not all(np.isfinite(rates)) or not all(0 <= r <= 1 for r in rates):
                    raise ValueError(f"Invalid rates: {key}")
                if rates[0] > rates[1] + 1e-8 or not np.isclose(rates[1] + rates[2], 1, atol=1e-8):
                    raise ValueError(f"Incoherent served/expired rates: {key}")
                totals = np.zeros(4, dtype=int)
                for order_type in ORDER_TYPES:
                    name = order_type.name
                    try:
                        n, on, served, expired = [int(row[f"{field}_{name}"]) for field in
                                                  ("released", "on_time_count", "served_count",
                                                   "expired_count")]
                    except (KeyError, ValueError) as exc:
                        raise ValueError(f"Missing/invalid per-type counts for {key}; "
                                         "obtain complete supplemental exports") from exc
                    if not 0 <= on <= served <= n or expired < 0 or served + expired != n:
                        raise ValueError(f"Incoherent {name} counts: {key}")
                    totals += n, on, served, expired
                if totals[0] <= 0 or not np.allclose(np.array(rates), totals[1:] / totals[0], atol=1e-8):
                    raise ValueError(f"Rates do not match outcome counts: {key}")
                rows[key] = row
    if rows.keys() != expected:
        raise ValueError(f"Incomplete comparison: {len(expected - rows.keys())} missing rows")
    for load in LOADS:
        for map_seed in maps:
            for day in days:
                cohort = [rows[m, load, map_seed, day] for m in methods
                          if (m, load, map_seed, day) in rows]
                for order_type in ORDER_TYPES:
                    if len({int(r[f"released_{order_type.name}"]) for r in cohort}) != 1:
                        raise ValueError("Entrants must see the same released orders")
    return rows


def grid(rows, methods, days, maps=(0,), metric="on_time", order_type=None, loads=LOADS):
    """Numerators/denominators [seed, load, map, day]; type ratios are pooled on resampling."""
    shape = (len(methods), len(loads), len(maps), len(days))
    num, den = np.empty(shape), np.ones(shape)
    for s, method in enumerate(methods):
        for l, load in enumerate(loads):
            for m, map_seed in enumerate(maps):
                for d, day in enumerate(days):
                    row = rows[method, load, map_seed, day]
                    if order_type:
                        num[s, l, m, d] = int(row[f"{metric}_count_{order_type}"])
                        den[s, l, m, d] = int(row[f"released_{order_type}"])
                    else:
                        num[s, l, m, d] = float(row[f"{'expiry' if metric == 'expired' else metric}_rate"])
    return num, den


def estimate(pair):
    num, den = pair
    denominator = den.sum(axis=(-2, -1))
    if np.any(denominator <= 0):
        raise ValueError("No orders for a reported type in a bootstrap sample")
    return float((num.sum(axis=(-2, -1)) / denominator).mean())


def bootstrap(a, b, n_boot=N_BOOT):
    """Paired days and independent matched-seed x day percentile CIs, fixed maps retained."""
    if a[0].shape != b[0].shape:
        # A deterministic reference has one row of seeds, repeated without new randomness.
        if b[0].shape[0] == 1 and a[0].shape[1:] == b[0].shape[1:]:
            b = tuple(np.repeat(x, a[0].shape[0], axis=0) for x in b)
        else:
            raise ValueError("Unmatched arm shapes")
    point = estimate(a) - estimate(b)
    n_seeds, _, _, n_days = a[0].shape
    outputs = {}
    for kind in ("paired_days", "two_way"):
        rng = np.random.default_rng(0)
        boot = np.empty(n_boot)
        for k in range(n_boot):
            ss = rng.integers(n_seeds, size=n_seeds) if kind == "two_way" else np.arange(n_seeds)
            dd = rng.integers(n_days, size=n_days)
            def take(pair):
                return tuple(x[ss][..., dd] for x in pair)
            boot[k] = estimate(take(a)) - estimate(take(b))
        outputs[kind] = [point, *map(float, np.percentile(boot, [2.5, 97.5]))]
    return outputs


def two_way_ci(differences, n_boot=N_BOOT, seed=0):
    """Compatibility helper for plain seed x day differences."""
    rng = np.random.default_rng(seed)
    ns, nd = differences.shape
    values = [differences[np.ix_(rng.integers(ns, size=ns), rng.integers(nd, size=nd))].mean()
              for _ in range(n_boot)]
    return [float(differences.mean()), *map(float, np.percentile(values, [2.5, 97.5]))]


def score(rows, methods, days=SELECT_SEEDS, maps=(0,), metric="on_time"):
    return estimate(grid(rows, methods, days, maps, metric, loads=LOADS[1:]))


def choose_reward(summaries):
    """Inputs are five-seed selection means; insertion-independent registered tie breaks."""
    v2 = summaries["v2"]
    eligible = [r for r in REWARD_ARMS if r in summaries
                and summaries[r]["on_time"] >= v2["on_time"] - .005 - 1e-12]
    return min(eligible, key=lambda r: (summaries[r]["expired"], list(REWARD_ARMS).index(r)))


def pilot_drop(score_difference, broken=False):
    return broken or score_difference < -.02 - 1e-12


def audit_pilot_logs(directory, arm):
    """Diagnostic summaries; do not turn arbitrary KL/curve thresholds into extra drop gates."""
    run = Path(directory) / f"{arm}_set_s11"
    with open(run / "updates.csv", newline="") as stream:
        updates = list(csv.DictReader(stream))
    with open(run / "evals.csv", newline="") as stream:
        evals = list(csv.DictReader(stream))
    if not updates or not evals:
        raise ValueError("Empty pilot logs")
    if int(updates[-1]["decisions"]) != 6000640 or int(evals[-1]["decisions"]) != 6000640:
        raise ValueError("Pilot did not complete its registered budget")
    diagnostic = {}
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        values = np.array([float(r[key]) for r in updates])
        if not np.isfinite(values).all():
            raise ValueError(f"Broken pilot: non-finite {key}")
        diagnostic[key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    curve = np.array([[float(r[f"on_time_{l}"]) for l in LOADS[1:]] for r in evals]).mean(1)
    if not np.isfinite(curve).all():
        raise ValueError("Broken pilot: non-finite selection curve")
    diagnostic["curve_first3"] = float(curve[:3].mean())
    diagnostic["curve_last3"] = float(curve[-3:].mean())
    diagnostic["curve_rising"] = diagnostic["curve_last3"] > diagnostic["curve_first3"]
    diagnostic["usual_range"] = (0 <= diagnostic["approx_kl"]["median"] <= .05
                                  and 0 <= diagnostic["clip_frac"]["median"] <= .3)
    return diagnostic


def selection_data(directory, arm, seeds):
    selected = entrants(load_json(selection_path(directory, arm)), arm, seeds)
    new = checkpoint_labels(selected)
    frozen = checkpoint_labels(frozen_paths(arm == "notable", seeds))
    maps = tuple(DEV_MAP_SEEDS) if arm == "notable" else (0,)
    rows = read_rows([export_path(directory, arm)], new + frozen + ["slack"], SELECT_SEEDS, maps)
    return rows, new, frozen, maps


def pilot(directory):
    result = {"survivors": [], "arms": {}, "selection_only": True}
    for arm in ARMS:
        # Invalid exports are NOT evidence of a broken policy. Missing logs or failed selection
        # identify a broken arm; malformed/partial evaluation data stop the analysis.
        try:
            diag = audit_pilot_logs(directory, arm)
            entrants(load_json(selection_path(directory, arm)), arm, (11,))
        except (OSError, ValueError, KeyError) as exc:
            result["arms"][arm] = {"survives": False, "broken": str(exc)}
            continue
        rows, new, frozen, maps = selection_data(directory, arm, (11,))
        diff = score(rows, new, maps=maps) - score(rows, frozen, maps=maps)
        survives = not pilot_drop(diff)
        result["arms"][arm] = {"survives": survives, "difference": diff, "diagnostics": diag}
        if survives:
            result["survivors"].append(arm)
    return result


def choose(directory, manifest, include_combined=False):
    survivors = manifest["survivors"]
    if not manifest.get("selection_only") or len(set(survivors)) != len(survivors) or set(survivors) - set(ARMS):
        raise ValueError("Invalid pilot manifest")
    summaries, completed = {}, []
    v2 = None
    for arm in survivors:
        rows, new, frozen, maps = selection_data(directory, arm, TRAINING_SEEDS)
        completed.append(arm)
        if arm == "notable":
            continue
        reference = {m: score(rows, frozen, metric=m) for m in ("on_time", "expired")}
        if v2 is not None and not all(np.isclose(v2[k], reference[k], atol=1e-12) for k in v2):
            raise ValueError("Frozen v2 selection results differ across exports")
        v2 = reference
        summaries[arm] = {m: score(rows, new, metric=m) for m in ("on_time", "expired")}
    if v2 is None:
        # Export a reference-only selection grid when all single-map pilots are dropped.
        labels = checkpoint_labels(frozen_paths())
        rows = read_rows([Path(directory) / "improvement_selection_frozen.csv"], labels + ["slack"], SELECT_SEEDS)
        v2 = {m: score(rows, labels, metric=m) for m in ("on_time", "expired")}
    reward_summaries = {"v2": v2, **{r: summaries[a] for r, a in REWARD_ARMS.items() if a in summaries}}
    reward = choose_reward(reward_summaries)
    helps = "dest" in summaries and summaries["dest"]["on_time"] >= v2["on_time"] + .005 - 1e-12
    combine = reward != "v2" and helps
    candidate = REWARD_ARMS[reward]
    if reward == "v2" and helps and summaries["dest"]["expired"] <= v2["expired"]:
        candidate = "dest"
    if include_combined and not combine:
        raise ValueError("Combined set was not triggered")
    if combine and include_combined:
        rows, new, _, _ = selection_data(directory, "combined", TRAINING_SEEDS)
        c = {m: score(rows, new, metric=m) for m in ("on_time", "expired")}
        selected_reward = reward_summaries[reward]
        if (c["on_time"] >= selected_reward["on_time"] + .005 - 1e-12
                and c["on_time"] >= v2["on_time"] - .005 - 1e-12
                and c["expired"] <= selected_reward["expired"]):
            candidate = "combined"
        summaries["combined"] = c
        completed.append("combined")
    paths = {a: entrants(load_json(selection_path(directory, a)), a) for a in completed}
    paths["frozen"] = frozen_paths()
    paths["mm"] = frozen_paths(True)
    return {"selection_only": True, "reward": reward, "destination_helps": helps,
            "combine": combine, "ready_for_fresh": not combine or include_combined,
            "candidate": candidate, "completed": completed, "paths": paths,
            "selection_means": {"v2": v2, **summaries}}


def export_selection(arm, selection, directory):
    seeds = (11,) if len(selection["runs"]) == 1 else TRAINING_SEEDS
    selected = entrants(selection, arm, seeds) if arm != "frozen" else []
    frozen = frozen_paths(arm == "notable", seeds)
    out = str(export_path(directory, arm))
    cmd = [sys.executable, "scripts/map_stress.py" if arm == "notable" else "scripts/evaluate.py",
           "--methods", "slack", "--checkpoints", *selected, *frozen,
           "--loads", ",".join(LOADS), "--days", "100", "--processes", "4", "--out", out]
    cmd += ["--maps", "1-10"] if arm == "notable" else ["--seeds", "select"]
    subprocess.run(cmd, check=True)


def validate_choice(choice):
    if not choice.get("selection_only") or not choice.get("ready_for_fresh"):
        raise ValueError("Freeze all selection decisions before fresh access")
    expected = set(choice["completed"]) | {"frozen", "mm"}
    if set(choice["paths"]) != expected or choice["candidate"] not in expected - {"mm"}:
        raise ValueError("Invalid frozen choice manifest")
    for arm, paths in choice["paths"].items():
        if len(paths) != len(TRAINING_SEEDS):
            raise ValueError("Every arm requires exactly five checkpoint paths")
        if arm in ("frozen", "mm"):
            if paths != frozen_paths(arm == "mm"):
                raise ValueError("Frozen entrant paths changed")
        else:
            entrants({"runs": {f"runs/{arm}_set_s{s}": {"selected": p}
                                for s, p in zip(TRAINING_SEEDS, paths)}}, arm)
    return {arm: checkpoint_labels(paths) for arm, paths in choice["paths"].items()}


def export_fresh(choice):
    validate_choice(choice)
    paths = list(dict.fromkeys(p for ps in choice["paths"].values() for p in ps))
    subprocess.run([sys.executable, "scripts/evaluate.py", "--methods", "", "--checkpoints", *paths,
                    "--loads", ",".join(LOADS), "--seeds", "fresh", "--days", "300",
                    "--processes", "4", "--out", "runs/improvement_fresh.csv"], check=True)


def lower_above(intervals, threshold):
    return all(ci[1] > threshold for ci in intervals.values())


def target_readings(on_time, served, expiry_reduction, horizon="horizon"):
    """Positive expiry_reduction means the candidate expires fewer than frozen v2."""
    superiority = {name: lower_above(ci, 0) for name, ci in on_time.items()}
    served_wins = {name: lower_above(ci, 0) for name, ci in served.items()}
    # The owner requires a point improvement >=1pp and uncertainty excluding zero,
    # rather than an unregistered requirement for the lower bound itself to exceed 1pp.
    expiry = all(ci[0] >= .01 - 1e-12 and ci[1] > 0 for ci in expiry_reduction.values())
    result = {"beat_all_on_time": all(superiority.values()), "on_time_by_reference": superiority,
              "served_by_reference": served_wins,
              "reach_horizon_with_0.5pp_tolerance": lower_above(on_time[horizon], -.005),
              "expiry_reduction_at_least_1pp": expiry}
    result["all_three"] = (result["beat_all_on_time"] and result["reach_horizon_with_0.5pp_tolerance"]
                           and result["expiry_reduction_at_least_1pp"])
    return result


def confirm(choice, paths, horizon="horizon", n_boot=N_BOOT):
    arms = validate_choice(choice)
    methods = [m for group in arms.values() for m in group] + ["slack", "rollout", "scenario16", horizon]
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate entrant labels")
    rows = read_rows(paths, methods, FRESH_SEEDS, horizon=horizon)
    candidate = choice["candidate"]
    groups = {**arms, "slack": ["slack"], "rollout": ["rollout"],
              "scenario16": ["scenario16"], horizon: [horizon]}
    result = {"candidate": candidate, "n_boot": n_boot, "rng_seed": 0,
              "scope": "post-locked-test exploratory confirmation", "absolute": {}, "contrasts": {},
              "per_seed": {}}
    primary = {metric: {} for metric in ("on_time", "served", "expired")}
    # All loads and types get both intervals. Only medium/high overall determines targets.
    for name, labels in groups.items():
        loads = LOADS[1:] if name == horizon else LOADS
        result["absolute"][name] = {}
        result["per_seed"][name] = {}
        result["contrasts"][name] = {}
        for load_group in [(l,) for l in loads] + [LOADS[1:]]:
            endpoint = "+".join(load_group)
            result["absolute"][name][endpoint] = {}
            result["contrasts"][name][endpoint] = {}
            result["per_seed"][name][endpoint] = {}
            for typ in [None] + [t.name for t in ORDER_TYPES]:
                type_key = typ or "all"
                for metric in primary:
                    key = f"{type_key}/{metric}"
                    b = grid(rows, labels, FRESH_SEEDS, metric=metric, order_type=typ, loads=load_group)
                    result["absolute"][name][endpoint][key] = estimate(b)
                    result["per_seed"][name][endpoint][key] = [estimate(tuple(x[s:s+1] for x in b))
                                                              for s in range(len(labels))]
                    if name == candidate:
                        continue
                    a = grid(rows, arms[candidate], FRESH_SEEDS, metric=metric,
                             order_type=typ, loads=load_group)
                    ci = bootstrap(a, b, n_boot)
                    result["contrasts"][name][endpoint][key] = ci
                    if load_group == LOADS[1:] and typ is None:
                        primary[metric][name] = ci
    # Preserve each registered factor contrast even when it is not the final candidate.
    result["factor_contrasts"] = {}
    for arm in choice["completed"]:
        reference = "mm" if arm == "notable" else "frozen"
        result["factor_contrasts"][arm] = {"reference": reference, "endpoints": {}}
        for load_group in [(l,) for l in LOADS] + [LOADS[1:]]:
            endpoint = "+".join(load_group)
            values = result["factor_contrasts"][arm]["endpoints"][endpoint] = {}
            for typ in [None] + [t.name for t in ORDER_TYPES]:
                for metric in primary:
                    a, b = [grid(rows, labels, FRESH_SEEDS, metric=metric, order_type=typ,
                                 loads=load_group) for labels in (arms[arm], arms[reference])]
                    values[f"{typ or 'all'}/{metric}"] = bootstrap(a, b, n_boot)
    if candidate == "frozen":
        reduction = {kind: [0., 0., 0.] for kind in ("paired_days", "two_way")}
    else:
        reduction = {kind: [-ci[0], -ci[2], -ci[1]] for kind, ci in primary["expired"]["frozen"].items()}
    result["units"] = {"absolute": "fraction", "contrasts": "fraction difference", "console": "percent and percentage points"}
    result["targets"] = target_readings(primary["on_time"], primary["served"], reduction, horizon)
    result["expiry_reduction_vs_frozen"] = reduction
    print_readings(result)
    for key in ("beat_all_on_time", "reach_horizon_with_0.5pp_tolerance", "expiry_reduction_at_least_1pp", "all_three"):
        print(f"{key}: {'PASS' if result['targets'][key] else 'FAIL'}")
    print("Failures are reported, not tuned away. All rates and intervals in JSON are fractions.")
    return result


def print_readings(result):
    """Human-readable percentages and percentage-point effects; JSON retains fractions."""
    for name, endpoints in result["absolute"].items():
        for endpoint, values in endpoints.items():
            for key, rate in values.items():
                line = f"{name} {endpoint} {key}: {100 * rate:.3f}%"
                ci = result["contrasts"][name][endpoint].get(key)
                if ci:
                    for kind, (point, lo, hi) in ci.items():
                        line += f"; candidate - reference {kind} {100*point:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] pp"
                print(line)
    for arm, factor in result.get("factor_contrasts", {}).items():
        for endpoint, values in factor["endpoints"].items():
            for key, intervals in values.items():
                line = f"factor {arm} - {factor['reference']} {endpoint} {key}"
                for kind, (point, lo, hi) in intervals.items():
                    line += f"; {kind} {100*point:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] pp"
                print(line)


def development_report(directory, n_boot=N_BOOT):
    rows, new, frozen, maps = selection_data(directory, "notable", TRAINING_SEEDS)
    result = {"scope": "C development; fixed maps 1-10, selection days", "absolute": {}, "contrasts": {}}
    for loads in [(l,) for l in LOADS] + [LOADS[1:]]:
        endpoint = "+".join(loads)
        result["absolute"][endpoint], result["contrasts"][endpoint] = {}, {}
        for typ in [None] + [t.name for t in ORDER_TYPES]:
            for metric in ("on_time", "served", "expired"):
                key = f"{typ or 'all'}/{metric}"
                a, b = [grid(rows, labels, SELECT_SEEDS, maps, metric, typ, loads) for labels in (new, frozen)]
                result["absolute"][endpoint][key] = {"new": estimate(a), "frozen_mm": estimate(b)}
                result["contrasts"][endpoint][key] = bootstrap(a, b, n_boot)
    result["per_map"] = {str(m): {metric: score(rows, new, maps=(m,), metric=metric)
                                                    - score(rows, frozen, maps=(m,), metric=metric)
                                         for metric in ("on_time", "served", "expired")} for m in maps}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("pilot", "choose", "export-selection", "export-fresh", "confirm", "development"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--arm", choices=ARMS + ("combined", "frozen"))
    parser.add_argument("--selection")
    parser.add_argument("--selection-dir", default="runs")
    parser.add_argument("--pilot-manifest")
    parser.add_argument("--include-combined", action="store_true")
    parser.add_argument("--choice")
    parser.add_argument("--csv", nargs="+")
    parser.add_argument("--horizon-label", default="horizon")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.export_selection:
        if not args.arm or (not args.selection and args.arm != "frozen"):
            parser.error("--export-selection requires --arm and --selection (except frozen)")
        export_selection(args.arm, load_json(args.selection) if args.selection else {"runs": {s: {} for s in TRAINING_SEEDS}}, args.selection_dir)
        return
    if args.export_fresh:
        export_fresh(load_json(args.choice))
        return
    if not args.out:
        parser.error("analysis modes require --out")
    if args.pilot:
        result = pilot(args.selection_dir)
    elif args.choose:
        result = choose(args.selection_dir, load_json(args.pilot_manifest), args.include_combined)
    elif args.development:
        result = development_report(args.selection_dir)
    else:
        if not args.csv or not args.choice:
            parser.error("--confirm requires --choice and --csv")
        result = confirm(load_json(args.choice), args.csv, args.horizon_label)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
