"""Render shipped PPO and Slack-aware on the same arrivals and simulated clock, the whole day.

The video: a title card, a four-step guide to the screen, the day side by side with running
charts (slower in demand peaks), pauses at moments picked from the two days' records, and the
final test's results as bars.
"""
import argparse
import csv
import json
from itertools import repeat
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EPS
from medical_uav.env.render import save_animation
from medical_uav.env.replay import day_record, decision_record, state_at
from medical_uav.env.video import TYPE_LABELS, DuelFrames, clock, duel_times
from scripts.demo import POLICY_NAMES, make_policy

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUMMARY = ROOT / "evidence/final_test/summary.json"
# final-test key, label, report colour, latency file and method name
METHODS = [("final", "PPO (ours)", "#e8590c", "networks", "checkpoints/entrants/f12_mm_e05_s11.pt"),
           ("horizon", "Rolling horizon", "#1f1f1f", "horizon", "horizon"),
           ("scenario16", "Scenario rollout", "#1c7ed6", "scenario16", "scenario16"),
           ("rollout", "Rollout", "#1c7ed6", "rollout", "rollout"),
           ("insertion", "Insertion", "#2b8a3e", "insertion", "insertion"),
           ("slack", "Slack-aware", "#7048e8", "slack", "slack")]
PAUSE_S = 8


def record_day(name, load, seed):
    """Keep the battery and visible slots alongside the ordinary decision facts."""
    policy, net = make_policy(name, seed)
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    decisions, done = [], False
    while not done:
        action = policy(env, obs, info["action_mask"])
        # the network's probabilities give the replay its top choices
        probs = net.probabilities(obs, info["action_mask"]) if net else None
        d = decision_record(env, action, probs)
        d.update(battery=env.sim.battery, visible=[o.id for o in env.slots])
        decisions.append(d)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return day_record(env, decisions, POLICY_NAMES[name])


# ---------- moments to pause at, each (time, title, two lines) ----------
def _outcome(o):
    if o["status"] == "expired":
        return "expired"
    return "on time" if o["end"] - o["release"] <= o["limit"] + EPS else "late"


def _name(r, o):
    nodes = r["map"]["nodes"]
    return (f"{TYPE_LABELS[o['type']].lower()} #{o['id']} "
            f"({nodes[o['from']]['name']} → {nodes[o['to']]['name']})")


def _pairs(ppo, rule, mine, other):
    """Orders whose outcome is `mine` under one side and `other` under the other, by time."""
    return sorted(((a, b) for a, b in zip(ppo["orders"], rule["orders"])
                   if _outcome(a) == mine and _outcome(b) == other),
                  key=lambda ab: ab[1]["end"] if mine == "on time" else ab[0]["end"])


def rule_failure(ppo, rule):
    """The first order the rule lets expire that PPO delivers on time."""
    a, b = _pairs(ppo, rule, "on time", "expired")[0]
    where = ("it was never picked up" if b["pickup"] is None else
             f"the rule picked it up at {clock(b['pickup'])} but did not deliver it in time")
    return (b["end"], f"{clock(b['end'])} · The rule loses {_name(rule, b)}",
            [f"It expires: {where}.",
             f"PPO delivered the same order on time at {clock(a['end'])}."])


def ppo_failure(ppo, rule):
    """The first order PPO lets expire that the rule delivers on time: PPO is not perfect."""
    a, b = sorted(((a, b) for a, b in zip(ppo["orders"], rule["orders"])
                   if _outcome(a) == "expired" and _outcome(b) == "on time"),
                  key=lambda ab: ab[0]["end"])[0]
    where = "never picked up" if a["pickup"] is None else "picked up but not delivered in time"
    return (a["end"], f"{clock(a['end'])} · PPO is not perfect: it loses {_name(ppo, a)}",
            [f"The order expires, {where}.",
             f"The rule delivered the same order on time at {clock(b['end'])}."])


def _waiting(r, t):
    return len(state_at(r, t)["waiting"])


def rule_charge(ppo, rule, waiting=_waiting):
    """The rule's first charge in a demand peak with at least five orders waiting."""
    peaks = rule["meta"]["peaks"]
    seg = next(s for s in rule["segments"] if s[0] == "charge" and waiting(rule, s[1]) >= 5
               and any(a <= s[1] < b for a, b in peaks))
    last = [s for s in ppo["segments"] if s[0] == "charge" and s[1] < seg[1]]
    lines = [f"Its battery is down to {seg[5]:.0f}%: it charges for {seg[2] - seg[1]:.0f} min "
             f"while {waiting(rule, seg[1])} orders wait."]
    lines.append(f"PPO last charged at {clock(last[-1][1])}, to {last[-1][6]:.0f}%, "
                 "and tops up in short stops through the day." if last else "")
    return seg[1], f"{clock(seg[1])} · The rule has to stop and charge", lines


def backlog_gap(ppo, rule):
    """In the last demand peak, the minute the rule's backlog exceeds PPO's the most."""
    a, b = ppo["meta"]["peaks"][-1]
    t = max(np.arange(a, b, 1.0), key=lambda m: _waiting(rule, m) - _waiting(ppo, m))
    sp, sr = state_at(ppo, t), state_at(rule, t)
    return (float(t), f"{clock(t)} · Second peak: {len(sr['waiting'])} orders wait for the rule, "
            f"{len(sp['waiting'])} for PPO",
            ["A longer backlog means more orders running out of time before pickup.",
             f"On time so far: PPO {sp['on_time']}, the rule {sr['on_time']}."])


def day_end(ppo, rule):
    p, r = ppo["meta"]["summary"], rule["meta"]["summary"]
    t = max(ppo["meta"]["end"], rule["meta"]["end"])
    line = lambda m: (f"{m['on_time']} on time, {m['delivered'] - m['on_time']} late, "
                      f"{m['expired']} expired")
    return (t, f"End of the day · the same {p['orders']} orders",
            [f"PPO: {line(p)}.   The rule: {line(r)}.",
             "One day is one example; the final test below averages 300 days."])


def phase_caption(meta, t):
    """The running caption: what to watch in this part of the day."""
    peaks = meta["peaks"]
    if t < peaks[0][0]:
        return ("Morning: few orders, both dispatchers keep up",
                ["Watch the map, the score above each side and the charts below.", ""], False)
    if any(a <= t < b for a, b in peaks):
        return ("Demand peak: orders arrive over three times as fast as outside the peaks",
                ["Watch the waiting squares on the maps and the backlog chart; "
                 "the video slows down here.", ""], False)
    if t < peaks[-1][0]:
        return ("Between the peaks: both work through the orders that are waiting", ["", ""],
                False)
    return "Afternoon: the remaining orders are cleared", ["", ""], False


# ---------- final test ----------
def serial_ms(file, method, load):
    """Mean serial decision time from evidence/latency."""
    with (ROOT / f"evidence/latency/{file}.csv").open() as f:
        return float(np.mean([float(r["ms"]) for r in csv.DictReader(f)
                              if r["method"] == method and r["load"] == load]))


def _ms(v):
    return f"{v:.2f} ms" if v < 10 else f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def final_rows(path=FINAL_SUMMARY, timed=True):
    """(label, colour, medium %, high %, decision time at high load) on the 30 held-out maps."""
    maps = json.loads(path.read_text())["maps"]
    return [(label, color, *(100 * maps[load]["means"][key][0] for load in ("medium", "high")),
             _ms(serial_ms(file, method, "high")) if timed else "")
            for key, label, color, file, method in METHODS]


TITLE = ["One drone serves a medical center and five hospitals. Orders for plasma,",
         "medicine, samples and supplies arrive all day, each with a deadline;",
         "the drone carries at most 3 boxes and must recharge on the way.", "",
         "Left: our PPO agent, a neural network trained by reinforcement learning.",
         "Right: Slack-aware, our best hand-written rule.",
         "Both get exactly the same orders on the same map.", "",
         "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seed", type=int, default=100007)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("outputs/compare.mp4"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    ppo, rule = [record_day(name, args.load, args.seed) for name in ("ppo", "slack")]
    r = DuelFrames(ppo, rule)
    fps = args.fps
    pauses = {t: (title, lines, True) for t, title, lines in
              (rule_charge(ppo, rule), rule_failure(ppo, rule), ppo_failure(ppo, rule),
               backlog_gap(ppo, rule), day_end(ppo, rule))}
    times = sorted(set(duel_times(ppo["meta"]["peaks"], r.end, fps)) | set(pauses))
    results = r.results("Final test: 30 held-out maps × 10 days per load", final_rows(),
                        "PPO: mean of five training runs. Time per decision: one CPU thread, "
                        "high load.",
                        "PPO is within about one point of the rolling-horizon planner at both loads"
                        " and decides thousands of times faster.")

    def frames():
        yield from repeat(r.card("Same orders, two dispatchers", TITLE,
                                 "ME5418 FINAL PROJECT · MEDICAL DELIVERY DRONE"), 10 * fps)
        start = ppo["meta"]["peaks"][0][0] + 30
        for step in range(len(r.GUIDE)):
            yield from repeat(r.guide(start, step), 5 * fps)
        for i, t in enumerate(times):
            caption = pauses.get(t) or phase_caption(ppo["meta"], t)
            yield from repeat(r.draw(t, caption), PAUSE_S * fps if t in pauses else 1)
            if i % (fps * 20) == 0:
                print(f"Rendering {clock(t)} ({i + 1}/{len(times)})", flush=True)
        yield from repeat(results, 14 * fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for k, (t, caption) in enumerate(sorted(pauses.items())):
        imageio.imwrite(args.out.with_name(f"{args.out.stem}_pause{k + 1}.png"), r.draw(t, caption))
    imageio.imwrite(args.out.with_name(f"{args.out.stem}_results.png"), results)
    n = save_animation(frames(), str(args.out), fps)
    print(f"Saved {args.out}: {n} frames, {n / fps:.1f} s")
    print("Pauses: " + ", ".join(clock(t) for t in sorted(pauses)))


if __name__ == "__main__":
    main()
