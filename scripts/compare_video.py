"""The group video: PPO and Slack-aware on the same high-load day, as a slide-style talk.

Title and outline; 1 the task, its map built up element by element; 2 how the PPO agent decides:
the network, then three of its decisions step by step in the replay's layout; 3 the whole day
side by side with the rule, with running charts and short pauses at moments picked from the
two records; 4 the final test's results; a summary.
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
from medical_uav.env.actions import decode
from medical_uav.env.config import EPS
from medical_uav.env.render import save_animation
from medical_uav.env.replay import day_record, decision_record, state_at
from medical_uav.env.video import (TYPE_LABELS, DuelFrames, ReplayFrames, clock, day_times)
from scripts.demo import POLICY_NAMES, make_policy

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUMMARY = ROOT / "evidence/final_test/summary.json"
NETWORK_FIG = ROOT / "docs/figures/M2_fig1_set_network.png"
# final-test key, label, report colour, latency file and method name
METHODS = [("final", "PPO (ours)", "#e8590c", "networks", "checkpoints/entrants/f12_mm_e05_s11.pt"),
           ("horizon", "Rolling horizon", "#1f1f1f", "horizon", "horizon"),
           ("scenario16", "Scenario rollout", "#1c7ed6", "scenario16", "scenario16"),
           ("rollout", "Rollout", "#1c7ed6", "rollout", "rollout"),
           ("insertion", "Insertion", "#2b8a3e", "insertion", "insertion"),
           ("slack", "Slack-aware", "#7048e8", "slack", "slack")]
OUTLINE = ["The task", "How our PPO agent decides", "One whole day: PPO vs a hand-written rule",
           "Results on 30 held-out maps"]
PAUSE_S, ANIM = 5, 4  # pause length (s); frames per step of the decision-card animation


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
    """The running caption: what is happening in this part of the day."""
    peaks = meta["peaks"]
    if t < peaks[0][0]:
        return "Morning: few orders, both keep up", [""], False
    if any(a <= t < b for a, b in peaks):
        return ("Demand peak: orders arrive more than three times as often",
                ["Watch the waiting lists grow and the backlog chart."], False)
    if t < peaks[-1][0]:
        return "Between the peaks: both work through the waiting orders", [""], False
    return "Afternoon: the last orders are cleared", [""], False


# ---------- part 2: three PPO decisions ----------
def _picked(d):
    kind, arg = decode(d["action"])
    return d["visible"][arg] if kind == "pick" else None


def _top(d):
    return ("The network scores every allowed action",
            f"Its top choice: {d['top'][0][0]}, {100 * d['top'][0][1]:.1f}%.")


def closeups(r):
    """Three decisions in the first demand peak, each (index, captions, focus, run until):
    a full batch unloaded at one stop, the most urgent pick among many waiting orders, and the
    charge that follows it (the network sure of both). Captions: situation, choices, the pick, the consequence."""
    a, b = r["meta"]["peaks"][0]
    ds, orders, nodes = r["decisions"], {o["id"]: o for o in r["orders"]}, r["map"]["nodes"]
    peak = [i for i, d in enumerate(ds) if a <= d["t"] < b]
    out = []

    i = next(i for i in peak if ds[i]["facts"][1].startswith("Frees 3 of 3"))
    d, s = ds[i], state_at(r, ds[i]["t"])
    box = [orders[k] for k in s["cargo"]]
    dest = nodes[box[0]["to"]]["name"]
    ok = all(_outcome(o) == "on time" for o in box)
    out.append((i, [(f"{clock(d['t'])} · Three boxes on board, all for {dest}",
                     f"{len(s['waiting'])} more orders are waiting; "
                     f"{d['facts'][0][0].lower() + d['facts'][0][1:]}."),
                    _top(d),
                    ("It takes the most likely action", f"One stop at {dest} delivers all three."),
                    ("The consequence", "All three arrive on time." if ok else
                     "Not all three arrive on time.")],
                {"nodes": (box[0]["to"],), "orders": tuple(s["cargo"])}, d["end"]))

    def urgent(i):
        s = state_at(r, ds[i]["t"])
        k = _picked(ds[i])
        return (k is not None and orders[k]["type"] == 0 and len(s["waiting"]) >= 10
                and k == min(s["waiting"], key=lambda j: orders[j]["deadline"])
                and ds[i]["top"][0][1] >= 0.95 and ds[i + 1]["text"].startswith("Charge"))
    i = next(i for i in peak if urgent(i))
    d, s = ds[i], state_at(r, ds[i]["t"])
    o = orders[_picked(d)]
    dest = nodes[o["to"]]["name"]
    mates = [orders[k] for k in s["cargo"] if orders[k]["to"] == o["to"]]
    also = f" Plasma #{mates[0]['id']} on board also goes to {dest}." if mates else ""
    out.append((i, [(f"{clock(d['t'])} · {len(s['waiting'])} orders are waiting",
                     f"Plasma #{o['id']} for {dest} is the most urgent: due in "
                     f"{o['deadline'] - d['t']:.0f} min.{also}"),
                    _top(d),
                    ("It takes the most likely action",
                     "The most urgent order, on the same trip as the box on board." if mates else
                     "The most urgent order.")],
                {"nodes": (o["to"],), "orders": (o["id"], *(m["id"] for m in mates))}, d["end"]))

    i += 1
    d, s = ds[i], state_at(r, ds[i]["t"])
    box = [orders[k] for k in s["cargo"]]
    kinds = " and ".join(sorted({TYPE_LABELS[o["type"]].lower() for o in box}))
    target = d["text"].rsplit(" ", 1)[1]
    done = max(o["end"] for o in box)
    ok = all(_outcome(o) == "on time" for o in box)
    out.append((i, [(f"{clock(d['t'])} · Battery {d['battery']:.0f}%, {len(box)} {kinds} "
                     f"box{'es' if len(box) > 1 else ''} on board", f"They are due in {min(o['deadline'] for o in box) - d['t']:.0f}"
                     " min at the earliest."),
                    _top(d),
                    ("It takes the most likely action",
                     f"It charges only to {target} ({d['facts'][1]}), not to 100%, then flies."),
                    ("The consequence", ("All boxes arrive on time: " if ok else "Arrivals: ") +
                     ", ".join(f"#{o['id']} at {clock(o['end'])}" for o in box) + ".")],
                {"nodes": (), "orders": tuple(s["cargo"])}, done))
    return out


def closeup_frames(view, i, captions, until, fps):
    """One decision step by step in the replay's layout, then its action at 2 simulated min/s."""
    d = view.r["decisions"][i]
    hold = lambda phase, cap, sec: repeat(view.draw(d["t"], phase, "", cap), round(sec * fps))
    yield from hold("Deciding", captions[0], 5)
    yield from hold("Deciding", captions[1], 4)
    yield from hold("Picked", captions[2], 3)
    m = round(0.7 * fps)
    for k in range(m):
        yield view.draw(d["t"], ("Moving", k / m), "", captions[2])
    last = captions[3] if len(captions) > 3 else captions[2]
    t = d["t"]
    while t < until - EPS:
        t = min(until, t + 2 / fps)
        yield view.draw(t, None, "", last)
    yield from repeat(view.draw(until, None, "", last), round(2 * fps))


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


TITLE = ["One drone, a medical center and five hospitals: which order next, and when to",
         "charge? We trained a neural network with reinforcement learning (PPO) to decide.", "",
         "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang"]
SLACK = ["Slack-aware is our best hand-written rule: earliest deadline first, unload where it is,"
         " carry orders going the same way together, hopeless orders last.",
         "We compare with it because it is the strongest rule we could write by hand; the "
         "planners appear in the results.",
         "Same map, same 66 orders, one shared clock. PPO on the left, the rule on the right."]


def summary_lines(rows):
    ppo, horizon, slack = rows[0], rows[1], rows[-1]
    return [f"PPO matches the strongest planner (rolling horizon) within about one point, and "
            f"decides in {ppo[4]} instead of {horizon[4]}.",
            f"At high load it delivers {ppo[3] - slack[3]:.1f} points more orders on time than "
            "our best hand-written rule.",
            "It still makes mistakes: on this day it let one sample expire that the rule "
            "delivered."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seed", type=int, default=100007)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("outputs/video.mp4"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    fps = args.fps
    ppo, rule = [record_day(name, args.load, args.seed) for name in ("ppo", "slack")]
    r = DuelFrames(ppo, rule)
    view = ReplayFrames(ppo, f"{args.load.capitalize()} load · demand seed {args.seed} · map 0",
                        scale=1.5)
    pauses = {t: (title, lines, True) for t, title, lines in
              (rule_charge(ppo, rule), ppo_failure(ppo, rule), backlog_gap(ppo, rule))}
    end_t, end_title, end_lines = day_end(ppo, rule)
    pauses[end_t] = (end_title, end_lines, True)
    rows = final_rows()
    first = closeups(ppo)[0][0]
    section = lambda k, note: repeat(r.outline(OUTLINE, k, note), 3 * fps)

    def frames():
        yield from repeat(r.card("Learning to dispatch a medical delivery drone", TITLE,
                                 "ME5418 FINAL PROJECT · GROUP 17"), 6 * fps)
        yield from repeat(r.outline(OUTLINE), 5 * fps)
        yield from section(0, "The world our dispatcher works in.")
        for k in range(len(r.TASK)):
            yield from repeat(r.task(k, ppo["decisions"][first]["t"]), round(3.5 * fps))
        yield from section(1, "What the network sees and how it picks an action.")
        yield from repeat(r.image_slide("2  How our PPO agent decides", NETWORK_FIG, [
            "It sees the drone (battery, place, cargo) and up to 12 waiting orders, one row each.",
            "Attention compares them; every allowed action gets a probability; it takes the top.",
            "Trained by PPO on 500 random maps for 12 million decisions."]), 8 * fps)
        for i, captions, focus, until in closeups(ppo):
            view.focus = focus
            yield from closeup_frames(view, i, captions, until, fps)
        view.focus = {"nodes": (), "orders": ()}
        yield from section(2, "The same day, side by side with our best hand-written rule.")
        yield from repeat(r.bullets("Our comparison: Slack-aware", SLACK, "3  ONE WHOLE DAY"),
                          7 * fps)
        for step in (0, 2, 4):  # map, waiting orders, running totals
            yield from repeat(r.guide(0.0, step), 3 * fps)
        times = sorted(set(day_times((ppo, rule), fps, r.end)) | set(pauses))
        seen, since = [None, None], [ANIM * 2, ANIM * 2]
        for n, t in enumerate(times):
            for k, rec in enumerate((ppo, rule)):
                i = state_at(rec, min(t, rec["meta"]["end"]))["decision"]
                since[k] = 0 if i != seen[k] else since[k] + 1
                seen[k] = i
            phases = tuple("Deciding" if c < ANIM else "Picked" if c < 2 * ANIM else None
                           for c in since)
            caption = pauses.get(t) or phase_caption(ppo["meta"], t)
            frame = r.draw(t, caption, phases, "time-lapse · 3-8 simulated min per second")
            yield from repeat(frame, (PAUSE_S + 2 * (t == end_t)) * fps if t in pauses else 1)
            if n % (fps * 20) == 0:
                print(f"Rendering {clock(t)} ({n + 1}/{len(times)})", flush=True)
        yield from section(3, "Six methods on 30 held-out maps, ten days each, two loads.")
        yield from repeat(r.results("Final test: 30 held-out maps × 10 days per load", rows,
                                    "PPO: mean of five training runs. Time per decision: one CPU "
                                    "thread, high load.",
                                    "PPO is within about one point of the rolling-horizon planner"
                                    " at both loads and decides thousands of times faster."),
                          12 * fps)
        yield from repeat(r.bullets("Summary", summary_lines(rows), "ME5418 GROUP 17"), 10 * fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = save_animation(frames(), str(args.out), fps)
    print(f"Saved {args.out}: {n} frames, {n / fps:.1f} s")


if __name__ == "__main__":
    main()
