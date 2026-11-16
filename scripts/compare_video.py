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
from medical_uav.env.replay import day_record, decision_record, short_text, state_at
from medical_uav.env.video import TYPE_LABELS, DuelFrames, ReplayFrames, clock, day_times
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
OUTLINE = ["The task", "How our PPO agent decides", "One whole day: PPO vs a hand-written rule",
           "Results on 30 new maps"]
PAUSE_S, ANIM = 7, 4  # pause length (s); frames per step of the decision-card animation


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
        d.update(battery=env.sim.battery, visible=[o.id for o in env.slots],
                 allowed=int(np.sum(info["action_mask"])))
        if net:  # the three most likely actions even when tiny, so the rest can be summarised
            d["top"] = [[short_text(env, int(k)), float(probs[k])]
                        for k in np.argsort(probs)[::-1][:min(3, d["allowed"])]]
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
    stops = lambda r: [s[2] - s[1] for s in r["segments"] if s[0] == "charge"]
    lines = [f"Its battery is down to {seg[5]:.0f}% and {waiting(rule, seg[1])} orders wait.",
             f"Over the whole day PPO makes {len(stops(ppo))} charging stops of "
             f"{np.mean(stops(ppo)):.1f} min on average, the rule {len(stops(rule))} of "
             f"{np.mean(stops(rule)):.1f} min."]
    return (seg[1], f"{clock(seg[1])} · The rule starts a {seg[2] - seg[1]:.0f}-minute charge",
            lines)


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
        return ("Morning: few orders, both keep up",
                ["Watch the gap in on-time deliveries and the waiting orders; "
                 "we pause at three moments."], False)
    if any(a <= t < b for a, b in peaks):
        return ("Demand peak: orders arrive more than three times as often",
                ["Watch the waiting lists grow and the backlog chart."], False)
    if t < peaks[-1][0]:
        return "Between the peaks: both work through the waiting orders", [""], False
    return "Afternoon: the remaining orders are delivered or expire", [""], False


# ---------- part 2: three PPO decisions ----------
def _picked(d):
    kind, arg = decode(d["action"])
    return d["visible"][arg] if kind == "pick" else None


def _top(d):
    return ("The network scores every allowed action",
            f"Of {d['allowed']} allowed actions, its top choice is {d['top'][0][0]} "
            f"({100 * d['top'][0][1]:.1f}%).")


# replay-layout boxes (1280 x 720) a close-up caption points at
MAP, CHOICES, ACTION = (24, 92, 520, 520), (906, 170, 358, 158), (562, 116, 700, 54)
CARGO, WAITING = (560, 332, 340, 288), (910, 332, 346, 288)


def closeups(r):
    """Three decisions in the first demand peak, each (index, captions, spots, focus, run until):
    a full batch unloaded at one stop, a plasma pick that shares the trip with a plasma box on
    board, and the charge that follows it (the network sure of each). Captions and spots: the
    situation, the network's choices, the pick, the consequence."""
    a, b = r["meta"]["peaks"][0]
    ds, orders, nodes = r["decisions"], {o["id"]: o for o in r["orders"]}, r["map"]["nodes"]
    peak = [i for i, d in enumerate(ds) if a <= d["t"] < b]
    out = []

    i = next(i for i in peak if ds[i]["facts"][1].startswith("Frees 3 of 3"))
    d, s = ds[i], state_at(r, ds[i]["t"])
    box = [orders[k] for k in s["cargo"]]
    dest = nodes[box[0]["to"]]["name"]
    ok = all(_outcome(o) == "on time" for o in box)
    out.append((i, [(f"{clock(d['t'])} · One stop, three boxes: all on board are for {dest}",
                     f"{len(s['waiting'])} more orders are waiting; "
                     f"{d['facts'][0][0].lower() + d['facts'][0][1:]}."),
                    _top(d),
                    ("It takes the most likely action", f"One stop at {dest} delivers all three."),
                    ("The consequence", "All three arrive on time." if ok else
                     "Not all three arrive on time.")],
                [[CARGO], [CHOICES], [ACTION, CHOICES], [MAP]],
                {"nodes": (box[0]["to"],), "orders": tuple(s["cargo"])}, d["end"]))

    def shared(i):
        s, k = state_at(r, ds[i]["t"]), _picked(ds[i])
        return (k is not None and orders[k]["type"] == 0 and len(s["waiting"]) >= 10
                and any(orders[c]["to"] == orders[k]["to"] for c in s["cargo"])
                and ds[i]["top"][0][1] >= 0.95 and ds[i + 1]["text"].startswith("Charge"))
    i = next(i for i in peak if shared(i))
    d, s = ds[i], state_at(r, ds[i]["t"])
    o = orders[_picked(d)]
    dest = nodes[o["to"]]["name"]
    mate = next(orders[c] for c in s["cargo"] if orders[c]["to"] == o["to"])
    out.append((i, [(f"{clock(d['t'])} · {len(s['waiting'])} orders are waiting",
                     f"Plasma #{o['id']} for {dest} is due in {o['deadline'] - d['t']:.0f} min; "
                     f"plasma #{mate['id']} on board also goes to {dest}."),
                    _top(d),
                    ("It takes the most likely action",
                     f"Plasma first, and on the same trip to {dest} as the box on board.")],
                [[WAITING, CARGO], [CHOICES], [ACTION, CHOICES]],
                {"nodes": (o["to"],), "orders": (o["id"], mate["id"])}, d["end"]))

    i += 1
    d, s = ds[i], state_at(r, ds[i]["t"])
    box = [orders[k] for k in s["cargo"]]
    kinds = " and ".join(sorted({TYPE_LABELS[o["type"]].lower() for o in box}))
    target = d["text"].rsplit(" ", 1)[1]
    seg = next(g for g in r["segments"] if g[0] == "charge" and g[1] >= d["t"] - EPS)
    rate = (seg[6] - seg[5]) / (seg[2] - seg[1])  # battery % per minute
    done = max(o["end"] for o in box)
    ok = all(_outcome(o) == "on time" for o in box)
    out.append((i, [(f"{clock(d['t'])} · Battery {d['battery']:.0f}%, {len(box)} {kinds} "
                     f"box{'es' if len(box) > 1 else ''} on board",
                     f"They are due in {min(o['deadline'] for o in box) - d['t']:.0f} min at the "
                     "earliest."),
                    _top(d),
                    ("It takes the most likely action",
                     f"It charges only to {target}: {seg[2] - seg[1]:.0f} min instead of about "
                     f"{(100 - seg[5]) / rate:.0f} min to 100%, then flies."),
                    ("The consequence", ("All boxes arrive on time: " if ok else "Arrivals: ") +
                     ", ".join(f"#{o['id']} at {clock(o['end'])}" for o in box) + ".")],
                [[CARGO], [CHOICES], [ACTION, CHOICES], [MAP]],
                {"nodes": (), "orders": tuple(s["cargo"])}, done))
    return out


def closeup_frames(view, i, captions, spots, until, fps):
    """One decision step by step in the replay's layout, each step's region spotlighted (fading
    in), then its action at 2 simulated min/s."""
    d = view.r["decisions"][i]
    fade = round(0.5 * fps)

    def hold(phase, cap, spot, sec):
        for k in range(fade):
            yield view.draw(d["t"], phase, "", cap, (spot, (k + 1) / fade))
        yield from repeat(view.draw(d["t"], phase, "", cap, (spot, 1.0)), round(sec * fps) - fade)
    yield from hold("Deciding", captions[0], spots[0], 5)
    yield from hold("Deciding", captions[1], spots[1], 5)
    yield from hold("Picked", captions[2], spots[2], 3)
    m = round(0.7 * fps)
    for k in range(m):
        yield view.draw(d["t"], ("Moving", k / m), "", captions[2], (spots[2], 1.0))
    last = captions[3] if len(captions) > 3 else captions[2]
    spot = (spots[3], 1.0) if len(spots) > 3 else None
    t = d["t"]
    while t < until - EPS:
        t = min(until, t + 2 / fps)
        yield view.draw(t, None, "", last, spot)
    yield from repeat(view.draw(until, None, "", last, spot), round(2 * fps))


# ---------- final test ----------
def serial_ms(file, method, load):
    """Mean serial decision time from evidence/latency."""
    with (ROOT / f"evidence/latency/{file}.csv").open() as f:
        return float(np.mean([float(r["ms"]) for r in csv.DictReader(f)
                              if r["method"] == method and r["load"] == load]))


def _ms(v):
    return f"{v:.2f} ms" if v < 10 else f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def final_rows(path=FINAL_SUMMARY, timed=True):
    """(label, colour, medium %, high %, decision time at high load) on the 30 new maps."""
    maps = json.loads(path.read_text())["maps"]
    return [(label, color, *(100 * maps[load]["means"][key][0] for load in ("medium", "high")),
             _ms(serial_ms(file, method, "high")) if timed else "")
            for key, label, color, file, method in METHODS]



TITLE = ["One drone, a medical center and five hospitals: which order next, and when to",
         "charge? We trained a neural network with reinforcement learning (PPO) to decide.", "",
         "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang"]
SLACK = ["Slack-aware is our best hand-written rule: earliest deadline first, unload where it is,"
         " carry orders going the same way together, hopeless orders last.",
         "It is simple enough to follow by eye; the stronger planning methods are compared in "
         "the final test.",
         "The day was chosen in advance: of 40 development days, the one with the median PPO "
         "lead over the rule.",
         "Same map, same 66 orders, one shared clock. PPO on the left, the rule on the right."]


def speedup(load="high"):
    """How many times faster PPO decides than the rolling horizon."""
    return serial_ms("horizon", "horizon", load) / serial_ms(METHODS[0][3], METHODS[0][4], load)


def summary_lines(rows, ppo, rule):
    p, h, slack = rows[0], rows[1], rows[-1]
    o = sorted((a for a, b in zip(ppo["orders"], rule["orders"])
                if _outcome(a) == "expired" and _outcome(b) == "on time"), key=lambda o: o["end"])[0]
    nodes = ppo["map"]["nodes"]
    lost = ("never went to " + nodes[o["from"]]["name"] + " for" if o["pickup"] is None else
            "did not deliver")
    return [f"On-time rates similar to the strongest planner (rolling horizon: {h[2]:.1f}% vs "
            f"PPO {p[2]:.1f}% at medium, {h[3]:.1f}% vs {p[3]:.1f}% at high load), with "
            f"decisions about {round(speedup(), -3):,.0f}× faster ({p[4]} vs {h[4]}).",
            f"At high load, PPO's on-time rate is {p[3] - slack[3]:.1f} points above our best "
            "hand-written rule (Slack-aware).",
            f"It is not perfect: on this day it {lost} {TYPE_LABELS[o['type']].lower()} "
            f"#{o['id']}, which expired; the rule delivered it on time."]


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
    view = ReplayFrames(ppo, f"Example day: {args.load} load on map 0, our milestone layout, "
                        "never used in training", scale=1.5)
    pauses = {t: (title, lines, True) for t, title, lines in
              (rule_charge(ppo, rule), ppo_failure(ppo, rule), backlog_gap(ppo, rule))}
    end_t, end_title, end_lines = day_end(ppo, rule)
    pauses[end_t] = (end_title, end_lines, True)
    rows = final_rows()
    first = closeups(ppo)[0][0]
    section = lambda k, note: repeat(r.outline(OUTLINE, k, note), 3 * fps)

    def frames():
        mark("title")
        yield from repeat(r.card("Learning to dispatch a medical delivery drone", TITLE,
                                 "ME5418 FINAL PROJECT · GROUP 17"), 6 * fps)
        mark("outline")
        yield from repeat(r.outline(OUTLINE), 5 * fps)
        mark("section 1")
        yield from section(0, "The world our dispatcher works in.")
        for k in range(len(r.TASK)):
            mark(f"task {k + 1}")
            yield from repeat(r.task(k, ppo["decisions"][first]["t"]), round(3.5 * fps))
        mark("section 2")
        yield from section(1, "What the network sees and how it picks an action.")
        mark("method")
        yield from repeat(r.method("2  How our PPO agent decides", [
            ("What it sees", "The drone (battery, place, cargo) and up to 12 waiting orders"),
            ("A neural network", "Attention compares every order with the others and with the "
                                 "drone"),
            ("A probability for each action", "Actions that are impossible now are masked out"),
            ("It takes the most likely one", "Then the next decision, again and again")], [
            "How it learned: PPO over 12 million decisions on 500 random maps. Reward: +1 for "
            "each on-time delivery, less for a late one, a penalty when an order expires.",
            f"In the next scenes each decision is slowed down for viewing; the network itself "
            f"needs {rows[0][4]}."]), 14 * fps)
        for i, captions, spots, focus, until in closeups(ppo):
            view.focus = focus
            mark(f"close-up {i}")
            yield from closeup_frames(view, i, captions, spots, until, fps)
        view.focus = {"nodes": (), "orders": ()}
        mark("section 3")
        yield from section(2, "The same day, side by side with our best hand-written rule.")
        mark("slack")
        yield from repeat(r.bullets("Our comparison: Slack-aware", SLACK, "3  ONE WHOLE DAY"),
                          12 * fps)
        steps = (0, 2, 4)  # map, waiting orders, running totals; shown at 10:30
        for n, step in enumerate(steps):
            mark(f"guide {n + 1}")
            yield from repeat(r.guide(150.0, step, f"{n + 1}/{len(steps)}",
                                      " (shown at 10:30; the day starts at 08:00)"), 5 * fps)
        times = sorted(set(day_times((ppo, rule), fps, r.end)) | set(pauses))
        mark("day")
        last_caption = None
        seen, since = [None, None], [ANIM * 2, ANIM * 2]
        for n, t in enumerate(times):
            for k, rec in enumerate((ppo, rule)):
                i = state_at(rec, min(t, rec["meta"]["end"]))["decision"]
                since[k] = 0 if i != seen[k] else since[k] + 1
                seen[k] = i
            phases = tuple("Deciding" if c < ANIM else "Picked" if c < 2 * ANIM else None
                           for c in since)
            caption = pauses.get(t) or phase_caption(ppo["meta"], t)
            if caption[0] != last_caption:
                mark(f"day: {caption[0]}")
                last_caption = caption[0]
            frame = r.draw(t, caption, phases, "time-lapse · 4-12 simulated min per second")
            yield from repeat(frame, (PAUSE_S + 2 * (t == end_t)) * fps if t in pauses else 1)
            if n % (fps * 20) == 0:
                print(f"Rendering {clock(t)} ({n + 1}/{len(times)})", flush=True)
        mark("section 4")
        yield from section(3, "Six methods, 30 new maps never used in training, ten days each.")
        mark("results")
        yield from repeat(r.results("Final test: 30 new maps never used in training, 300 days "
                                    "per load", rows, "Mean on-time rate; PPO: mean of five "
                                    "training runs. Time per decision: one CPU thread, high load.",
                                    f"PPO {rows[0][2]:.1f}% vs rolling horizon {rows[1][2]:.1f}% "
                                    f"(medium), {rows[0][3]:.1f}% vs {rows[1][3]:.1f}% (high); "
                                    f"about {round(speedup(), -3):,.0f}× faster per decision."),
                          18 * fps)
        mark("summary")
        yield from repeat(r.bullets("Summary", summary_lines(rows, ppo, rule), "ME5418 GROUP 17"),
                          12 * fps)

    marks, count = [], [0]

    def mark(name):  # where each part starts, for review copies with subtitles
        marks.append((name, count[0] / fps))

    def counted(frames):
        for frame in frames:
            count[0] += 1
            yield frame

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = save_animation(counted(frames()), str(args.out), fps)
    args.out.with_suffix(".json").write_text(json.dumps(marks, indent=0))
    print(f"Saved {args.out}: {n} frames, {n / fps:.1f} s")


if __name__ == "__main__":
    main()
