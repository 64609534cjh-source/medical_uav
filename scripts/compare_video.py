"""Render shipped PPO and Slack-aware on the same arrivals and simulated clock."""
import argparse
import csv
from itertools import repeat
from pathlib import Path
from textwrap import wrap

import imageio.v2 as imageio
import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import decode
from medical_uav.env.render import save_animation
from medical_uav.env.replay import day_record, decision_record, state_at
from medical_uav.env.video import ComparisonFrames, TYPE_LABELS, clock, compare_times
from scripts.demo import POLICY_NAMES, make_policy

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "runs/m3_screen_eval_all.csv"
SCREEN_FALLBACK = ROOT.parent / "medical_uav/runs/m3_screen_eval_all.csv"
SHIPPED_METHOD = "m3_set_s11/ckpt_06000640"


def record_day(name, load, seed):
    """Keep the battery and visible slots alongside the ordinary decision facts."""
    policy, _ = make_policy(name, seed)
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    decisions, done = [], False
    while not done:
        action = policy(env, obs, info["action_mask"])
        d = decision_record(env, action)
        d.update(battery=env.sim.battery, visible=[o.id for o in env.slots])
        decisions.append(d)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return day_record(env, decisions, POLICY_NAMES[name])


def decision_line(label, d):
    return (f"{label} decision at {clock(d['t'])}: {d['short']} · "
            f"battery {d['battery']:.0f}% · {d['facts'][-1]}")


def pauses(left, right, start, end):
    """First charge and first distinct service action, selected without outcome filtering."""
    found = {}
    for d in left["decisions"]:
        if not start <= d["t"] <= end:
            continue
        kind, _ = decode(d["action"])
        s = state_at(right, d["t"])
        other = right["decisions"][s["decision"]]
        if kind == "charge" and "charge" not in found:
            label = "Charging decision"
        elif kind in ("pick", "deliver") and "service" not in found:
            if d["text"] == other["text"]:
                continue
            label = "Service decision"
        else:
            continue
        key = "charge" if kind == "charge" else "service"
        note = [f"PAUSED · {label} · {clock(d['t'])}", decision_line("PPO", d),
                decision_line("Slack", other), f"PPO: {d['facts'][0]}"]
        found[key] = (d["t"], note)
    return sorted(found.values())


def development_summary(path=None):
    """Mean daily rates on the 40 screen days, never presented as locked-test results."""
    if path is None:
        path = SCREEN if SCREEN.exists() else SCREEN_FALLBACK
    if not path.exists():
        return []
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["load"] == "high"]
    means = [np.mean([float(r["on_time_rate"]) for r in rows if r["method"] == method])
             for method in (SHIPPED_METHOD, "slack")]
    days = sum(r["method"] == SHIPPED_METHOD for r in rows)
    return [f"{days} high-load screen days: PPO {100 * means[0]:.1f}% vs "
            f"Slack {100 * means[1]:.1f}% on time", "Development results; not the locked test."]


def failure_lines(left, right):
    """Show the first PPO failure by resolution time, with order ID breaking ties."""
    failures = [o for o in left["orders"] if o["status"] == "expired" or
                (o["end"] is not None and o["end"] > o["deadline"])]
    if not failures:
        return ["No late or expired PPO orders on this day."]
    o = min(failures, key=lambda o: (o["end"], o["id"]))
    other = next(p for p in right["orders"] if p["id"] == o["id"])
    nodes = left["map"]["nodes"]
    route = f"{nodes[o['from']]['name']} → {nodes[o['to']]['name']}"
    outcome = "expired without pickup" if o["pickup"] is None else (
        "delivered late" if o["status"] == "delivered" else "expired on board")
    other_outcome = other["status"]
    if other_outcome == "delivered":
        other_outcome += " on time" if other["end"] <= other["deadline"] else " late"
    return [f"One PPO failure: #{o['id']} {TYPE_LABELS[o['type']]} · {route}",
            f"Due {clock(o['deadline'])}; {outcome} at {clock(o['end'])}.",
            f"Same order under Slack: {other_outcome} at {clock(other['end'])}."]


def end_lines(left, right, summary):
    lines = [f"Whole day · {left['meta']['summary']['orders']} identical arrivals", ""]
    for label, record in (("PPO", left), ("Slack", right)):
        m = record["meta"]["summary"]
        lines.append(f"{label}: {m['on_time']} on time · {m['delivered'] - m['on_time']} late · "
                     f"{m['expired']} expired · {m['km']:.1f} km")
    lines += [""] + failure_lines(left, right) + [""]
    return lines + [part for line in summary for part in wrap(line, width=90)]


def intro_lines(load, seed, start, end):
    return [f"{load.capitalize()} load · demand seed {seed} · {clock(start)}–{clock(end)}",
            "One clock. Identical arrivals. Two independent dispatchers.",
            "Watch routes, deadlines, cargo and battery; two decisions pause.", "",
            "Default day: nearest the median PPO-minus-Slack on-time difference",
            "on high-load screen days; lowest day ID breaks ties.",
            "Flights: 1 simulated min/s; both grounded: 8 min/s."]


def video_frames(renderer, times, holds, fps, intro, endings):
    yield from repeat(intro, 4 * fps)
    notes = dict(holds)
    for i, t in enumerate(times):
        note = notes.get(t)
        frame = renderer.draw(t, note)
        yield from repeat(frame, 7 * fps if note else 1)
        if i % (fps * 20) == 0:
            print(f"Rendering {clock(t)} ({i + 1}/{len(times)} frames)", flush=True)
    for card in endings:
        yield from repeat(card, 8 * fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seed", type=int, default=100004)
    parser.add_argument("--start", type=float, default=50)
    parser.add_argument("--end", type=float, default=200)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--summary", type=Path,
                        help="UTF-8 aggregate-result lines for the end card")
    parser.add_argument("--out", type=Path, default=Path("outputs/compare.mp4"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    left, right = [record_day(name, args.load, args.seed) for name in ("ppo", "slack")]
    renderer = ComparisonFrames(left, right)
    holds = pauses(left, right, args.start, args.end)
    times = sorted(set(compare_times(left, right, args.fps, args.start, args.end)) |
                   {t for t, _ in holds})
    summary = (args.summary.read_text(encoding="utf-8").splitlines() if args.summary else
               development_summary())
    lines = end_lines(left, right, summary)
    endings = [renderer.card("The whole day, including a failure", lines[i:i + 12])
               for i in range(0, len(lines), 12)]
    intro = renderer.card("Same orders, two dispatchers",
                          intro_lines(args.load, args.seed, args.start, args.end))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    peak = next(((max(args.start, a) + min(args.end, b)) / 2 for a, b in left["meta"]["peaks"]
                 if max(args.start, a) < min(args.end, b)), args.start)
    stills = {"peak": renderer.draw(peak), "end": endings[0]}
    if holds:
        stills["paused"] = renderer.draw(*holds[0])
    for name, frame in stills.items():
        imageio.imwrite(args.out.with_name(f"{args.out.stem}_{name}.png"), frame)
    frames = video_frames(renderer, times, holds, args.fps, intro, endings)
    n = save_animation(frames, str(args.out), args.fps)
    print(f"Saved {args.out}: {n} frames, {n / args.fps:.2f} seconds")
    print(f"Pauses: {', '.join(clock(t) for t, _ in holds)}")


if __name__ == "__main__":
    main()
