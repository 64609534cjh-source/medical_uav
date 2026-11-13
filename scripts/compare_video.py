"""Render shipped PPO and Slack-aware on the same arrivals and simulated clock."""
import argparse
import csv
import json
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
FINAL_SUMMARY = ROOT / "evidence/final_test/summary.json"
SUMMARY_ROWS = [("final", "PPO (ours, mean of 5 runs)"), ("horizon", "Rolling horizon"),
                ("scenario16", "Scenario rollout"), ("rollout", "Rollout"), ("insertion", "Insertion"),
                ("slack", "Slack-aware")]
TIMED = {"PPO": ("networks", "checkpoints/entrants/f12_mm_e05_s11.pt"), "horizon": ("horizon", "horizon")}


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


def final_test_summary(path=FINAL_SUMMARY):
    """Mean daily on-time rates on the final test's 30 unseen maps, medium / high load."""
    if not path.exists():
        return []
    maps = json.loads(path.read_text())["maps"]
    lines = ["Final test, 30 unseen maps x 10 days, on time at medium / high load:"]
    for key, label in SUMMARY_ROWS:
        rates = [100 * maps[load]["means"][key][0] for load in ("medium", "high")]
        lines.append(f"{label}: {rates[0]:.1f}% / {rates[1]:.1f}%")
    ms = {name: [serial_ms(file, method, load) for load in ("medium", "high")]
          for name, (file, method) in TIMED.items()}
    if all(all(v) for v in ms.values()):
        lines += ["", "Mean time per decision on one CPU thread, medium / high load:",
                  f"PPO {ms['PPO'][0]:.2f} / {ms['PPO'][1]:.2f} ms · "
                  f"rolling horizon {ms['horizon'][0]:,.0f} / {ms['horizon'][1]:,.0f} ms"]
    return lines


def serial_ms(file, method, load):
    """Mean serial decision time from evidence/latency, or None if absent."""
    path = ROOT / f"evidence/latency/{file}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        values = [float(r["ms"]) for r in csv.DictReader(f) if r["method"] == method and r["load"] == load]
    return float(np.mean(values)) if values else None


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


TITLE_LINES = ["One drone serves a medical center (M) and five hospitals (H1-H5).",
               "Orders for plasma, medicine, samples and supplies arrive all day, each with a deadline;",
               "the drone carries at most 3 boxes and must recharge at C1 or C2.", "",
               "Left: our PPO agent, a set network with attention trained by reinforcement learning.",
               "Right: Slack-aware, our best hand-written rule (earliest deadline first, local",
               "unloading and batching, hopeless orders last).", "",
               "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang"]


def intro_lines(load, seed, start, end):
    return [f"{load.capitalize()} load · demand seed {seed} · map 0 · {clock(start)}–{clock(end)}",
            "One clock. Identical arrivals. Two independent dispatchers.",
            "Watch routes, deadlines, cargo and battery; two decisions pause to show both choices.", "",
            "The day was chosen in advance: among 40 development days, PPO's lead over",
            "Slack-aware is closest to the median on this one.",
            "Flights: 1 simulated min/s; both grounded: 8 min/s."]


def video_frames(renderer, times, holds, fps, intro, endings):
    for card, seconds in intro:
        yield from repeat(card, seconds * fps)
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
    parser.add_argument("--seed", type=int, default=100007)
    parser.add_argument("--start", type=float, default=40)
    parser.add_argument("--end", type=float, default=190)
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
               final_test_summary())
    endings = [renderer.card("The whole day, including a failure", end_lines(left, right, []))]
    if summary:
        endings.append(renderer.card("Final test: all methods", summary, kicker="RESULTS"))
    intro = [(renderer.card("Learning to dispatch a medical delivery drone", TITLE_LINES,
                            kicker="ME5418 FINAL PROJECT"), 10),
             (renderer.card("Same orders, two dispatchers",
                            intro_lines(args.load, args.seed, args.start, args.end)), 7)]
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
