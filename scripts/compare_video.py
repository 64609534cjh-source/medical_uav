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
from matplotlib import font_manager

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import decode
from medical_uav.env.config import EPS
from medical_uav.env.render import save_animation
from medical_uav.env.replay import day_record, decision_record, short_text, state_at
from medical_uav.env.video import (TYPE_LABELS, DuelFrames, ReplayFrames, Subtitles, clock,
                                   day_times, zoom)
from scripts.demo import POLICY_NAMES, make_policy

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUMMARY = ROOT / "evidence/final_test/summary.json"
# final-test key, label, report colour, latency file and method name
METHODS = [("final", "PPO (ours)", "#e8590c", "networks", "checkpoints/entrants/f12_mm_e05_s11.pt"),
           ("horizon", "Rolling horizon", "#1f1f1f", "horizon", "horizon"),
           ("scenario16", "Scenario rollout", "#1c7ed6", "scenario16", "scenario16"),
           ("rollout", "Rollout", "#1c7ed6", "rollout", "rollout"),
           ("insertion", "Insertion", "#2b8a3e", "insertion", "insertion"),
           ("slack", "Slack-aware", "#7048e8", "slack", "slack"),
           ("nearest", "Nearest-first", "#868e96", "rules", "nearest"),
           ("edf", "EDF", "#868e96", "rules", "edf"),
           ("fcfs", "FCFS", "#868e96", "rules", "fcfs"),
           ("random", "Random", "#868e96", "rules", "random")]
OUTLINE = ["The delivery task", "How PPO chooses actions", "One full day: PPO vs the Slack-aware rule",
           "Ten methods on 30 new maps"]
PAUSE_S, ANIM_S = 7, 0.33  # pause length (s); seconds per step of the decision-card animation
FADE, FADE_S = "fade", 0.4  # marker in a frame stream for a dissolve, and its length (s)


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


def _lost(ppo, rule):
    """The first order PPO lets expire that the rule delivers on time: (PPO's, the rule's)."""
    return sorted(((a, b) for a, b in zip(ppo["orders"], rule["orders"])
                   if _outcome(a) == "expired" and _outcome(b) == "on time"),
                  key=lambda ab: ab[0]["end"])[0]


def _cap(text):
    return text[:1].upper() + text[1:]


def _visit(r, node, a, b):
    """When the drone first reaches node between times a and b, or None."""
    return next((s[2] for s in r["segments"] if s[0] == "fly" and s[4] == node
                 and a <= s[2] < b), None)


def ppo_failure(ppo, rule):
    """The first order PPO lets expire that the rule delivers on time: PPO is not perfect."""
    a, b = _lost(ppo, rule)
    place = ppo["map"]["nodes"][a["from"]]["name"]
    seen = _visit(ppo, a["from"], a["release"], a["end"])
    why = (f"PPO was at {place} at {clock(seen)} for another box but did not take this one."
           if seen is not None and a["pickup"] is None else
           f"PPO never went to {place} while it waited." if a["pickup"] is None else
           "PPO picked it up but did not deliver it in time.")
    return (a["end"], f"{clock(a['end'])} · {_cap(_name(ppo, a))} expires: PPO never "
            "delivered it",
            [f"{why} Slack-aware picked it up at {clock(b['pickup'])} and delivered it on time at "
             f"{clock(b['end'])}."])


def _waiting(r, t):
    return len(state_at(r, t)["waiting"])


def _per_stop(r):
    """Boxes delivered per delivery stop over the day."""
    stops = [d for d in r["decisions"] if decode(d["action"])[0] == "deliver"]
    return sum(1 for o in r["orders"] if o["status"] == "delivered") / len(stops)


def backlog_gap(ppo, rule):
    """In the last demand peak, the minute the rule's backlog exceeds PPO's the most, with the
    day's boxes per delivery stop and flying distance."""
    a, b = ppo["meta"]["peaks"][-1]
    t = max(np.arange(a, b, 1.0), key=lambda m: _waiting(rule, m) - _waiting(ppo, m))
    sp, sr = state_at(ppo, t), state_at(rule, t)
    km = lambda r: r["meta"]["summary"]["km"]
    return (float(t), f"{clock(t)} · Orders awaiting pickup: {len(sr['waiting'])} for "
            f"Slack-aware, {len(sp['waiting'])} for PPO",
            [f"Part of the reason, over the whole day: PPO delivers {_per_stop(ppo):.1f} boxes per "
             f"delivery stop (Slack-aware {_per_stop(rule):.1f}) and flies {km(ppo):.0f} km "
             f"(Slack-aware {km(rule):.0f} km), so its drone has more time for new orders."])


def day_end(ppo, rule):
    p, r = ppo["meta"]["summary"], rule["meta"]["summary"]
    t = max(ppo["meta"]["end"], rule["meta"]["end"])
    line = lambda m: (f"{m['on_time']} on time, {m['delivered'] - m['on_time']} late, "
                      f"{m['expired']} expired")
    return (t, f"End of the day · final outcomes for the same {p['orders']} orders",
            [f"PPO: {line(p)}. Slack-aware: {line(r)}. This is one example day; the final test "
             "averages 300 days per demand level."])


def phase_caption(meta, t):
    """The running caption: what is happening in this part of the day."""
    peaks = meta["peaks"]
    if t < peaks[0][0]:
        return ("Morning: few orders, both keep up",
                ["Watch the gap in on-time deliveries and in orders awaiting pickup. We pause at "
                 "three moments."], False)
    if any(a <= t < b for a, b in peaks):
        return ("Demand peak: orders arrive about 3.3 times as often as outside the peaks",
                ["Watch the lists of orders awaiting pickup and the backlog chart."], False)
    if t < peaks[-1][0]:
        return "Between the peaks: both work through the orders awaiting pickup", [""], False
    return (f"After the second peak: orders keep arriving until {clock(meta['release_window'])}",
            ["Both keep working until every order is delivered or has expired."], False)


# ---------- part 2: two stretches of PPO's day, decision by decision ----------
def _picked(d):
    kind, arg = decode(d["action"])
    return d["visible"][arg] if kind == "pick" else None


def _label(o, nodes):
    return f"{TYPE_LABELS[o['type']].lower()} #{o['id']} for {nodes[o['to']]['name']}"


def _delivered(r, d):
    """Orders delivered by decision d."""
    return sorted((o for o in r["orders"] if o["status"] == "delivered"
                   and d["t"] < o["end"] <= d["end"] + EPS), key=lambda o: o["end"])


def _urgent(r, d, orders):
    s = state_at(r, d["t"])
    return min(s["waiting"], key=lambda j: orders[j]["deadline"]) if s["waiting"] else None


def _deliver_line(r, d, nodes, keep=()):
    """What a delivery unloads, and which of the boxes in `keep` stay on board."""
    got = _delivered(r, d)
    n = len(got)
    line = (f"Delivers {n} box{'es' if n > 1 else ''} at {nodes[got[0]['to']]['name']}, " +
            ("on time." if n == 1 and _outcome(got[0]) == "on time" else
             "all on time." if all(_outcome(o) == "on time" for o in got) else "not all on time."))
    kept = [o for o in r["orders"] if o["id"] in keep and o["id"] in state_at(r, d["end"])["cargo"]]
    if kept:
        line += (" Still on board: " + " and ".join(
            f"{TYPE_LABELS[o['type']].lower()} #{o['id']}" for o in kept) +
            f" for {nodes[kept[0]['to']]['name']}.")
    return line


def stretches(r):
    """Two stretches of the first demand peak, each (title, intro, steps, focus, until, plan,
    shows) with steps (decision index, subtitle title, subtitle line):
    1. after a delivery it takes a box from there instead of flying on empty, and collects boxes
       for one hospital at three places, delivering them in one visit;
    2. it lets the order with the earliest deadline (a plasma) wait for a short nearby job, then
       carries two plasma boxes on one trip after a short charge; both are still on time."""
    a, b = r["meta"]["peaks"][0]
    ds, orders, nodes = r["decisions"], {o["id"]: o for o in r["orders"]}, r["map"]["nodes"]
    name = lambda k: nodes[k]["name"]
    peak = [i for i, d in enumerate(ds) if a <= d["t"] < b]
    head = lambda d: f"{clock(d['t'])} decision · {d['text']}"
    out = []

    end = next(i for i in peak if ds[i]["facts"][1].startswith("Unloads 3 of the 3"))
    boxes = [orders[k] for k in state_at(r, ds[end]["t"])["cargo"]]
    target = boxes[0]["to"]
    first = min(i for i in range(end) if _picked(ds[i]) in {o["id"] for o in boxes})
    start = max(j for j in range(first + 1) if not state_at(r, ds[j]["t"])["cargo"])
    order = [name(o["from"]) for o in sorted(boxes, key=lambda o: o["pickup"])]
    places = ", ".join(order[:-1]) + " and " + order[-1]
    steps, n = [], 0
    for i in range(start, end + 1):
        d = ds[i]
        k = _picked(d)
        if k is not None:
            o, u = orders[k], _urgent(r, d, orders)
            line = (f"{TYPE_LABELS[o['type']]} #{k} at {name(o['from'])} for {name(o['to'])}, "
                    f"deadline {clock(o['deadline'])}.")
            if i == start:
                line = (f"After delivering at {name(o['from'])}, it takes {_label(o, nodes)} from "
                        "there, so the next flight is not empty.")
            if o["to"] == target:
                n += 1
                line += f" Box {n} of 3 for {name(target)}."
            if u is not None and u != k:
                line += (f" {_cap(_label(orders[u], nodes))} has an earlier deadline "
                         f"({clock(orders[u]['deadline'])}).")
        elif i == end:
            line = (f"It delivers the three boxes collected at {places} in one visit to "
                    f"{name(target)}" + (", all on time." if all(_outcome(o) == "on time"
                                                                  for o in boxes) else "."))
        else:
            line = _deliver_line(r, d, nodes, keep={o["id"] for o in boxes})
        steps.append((i, head(d), line))
    here = name(orders[_picked(ds[start])]["from"])
    out.append((f"Collecting three boxes for {name(target)} along the way",
                f"The highlighted boxes all go to {name(target)}: PPO collects them at {places}, "
                "then delivers them in one visit.", steps,
                {"nodes": (target,), "orders": tuple(o["id"] for o in boxes)}, ds[end]["end"],
                [f"When: simulated time {clock(ds[start]['t'])}–{clock(ds[end]['end'])}, during "
                 "the first demand peak (a period with more orders).",
                 f"What happens: after a delivery at {here} the drone takes a sample from there "
                 f"instead of flying on empty, and it collects three boxes for {name(target)} at "
                 f"{places}.",
                 f"Watch: the highlighted boxes for {name(target)}: where each one is picked up, "
                 "and which boxes stay on board between stops."],
                "What this example shows: PPO plans whole trips, not single orders. Boxes for the "
                "same place are collected along the way and delivered together."))

    i = next(i for i in peak if i > end and _picked(ds[i]) is not None
             and _urgent(r, ds[i], orders) != _picked(ds[i])
             and orders[_urgent(r, ds[i], orders)]["type"] == 0
             and _outcome(orders[_urgent(r, ds[i], orders)]) == "on time")
    u = orders[_urgent(r, ds[i], orders)]
    last = next(j for j in range(i, len(ds)) if u in _delivered(r, ds[j]))
    back = next(dd["t"] for dd in ds[i + 1:] if _picked(dd) == u["id"])
    left = u["deadline"] - ds[i]["t"]
    o0 = orders[_picked(ds[i])]
    steps = []
    for j in range(i, last + 1):
        d = ds[j]
        k = _picked(d)
        kind = decode(d["action"])[0]
        if j == i:
            line = (f"Plasma #{u['id']} waits at {name(u['from'])} for {name(u['to'])}, deadline "
                    f"{clock(u['deadline'])}, {left:.0f} min from now. PPO first takes "
                    f"{_label(o0, nodes)}, waiting here; {back - d['t']:.0f} min later it heads "
                    "for the plasma.")
        elif k is not None:
            o = orders[k]
            mates = [c for c in state_at(r, d["t"])["cargo"] if orders[c]["to"] == o["to"]]
            line = (f"{TYPE_LABELS[o['type']]} #{k} at {name(o['from'])} for {name(o['to'])}, "
                    f"deadline {clock(o['deadline'])}.")
            if mates:
                line += (f" Plasma #{mates[0]} for {name(o['to'])} is already on board, so one "
                         "trip serves both.")
        elif kind == "charge":
            seg = next(g for g in r["segments"] if g[0] == "charge" and g[1] >= d["t"] - EPS)
            rate = (seg[6] - seg[5]) / (seg[2] - seg[1])  # battery % per minute
            line = (f"Battery {seg[5]:.0f}%: charging to {seg[6]:.0f}% takes "
                    f"{seg[2] - seg[1]:.0f} min; charging to 100% would take about "
                    f"{(100 - seg[5]) / rate:.0f} min.")
        elif j == last:
            got = _delivered(r, d)
            line = ("Delivered before their deadlines: " if all(
                _outcome(o) == "on time" for o in got) else "Delivered: ") + "; ".join(
                f"plasma #{o['id']} at {clock(o['end'])}, {60 * (o['deadline'] - o['end']):.0f} s "
                "early" for o in got) + "."
        else:
            line = _deliver_line(r, d, nodes)
        steps.append((j, head(d), line))
    done = max(o["end"] for o in _delivered(r, ds[last]))
    out.append((f"Nearby work before an urgent plasma",
                f"At {clock(ds[i]['t'])}, plasma #{u['id']} has the earliest deadline of all "
                "orders awaiting pickup. Watch what PPO does first.", steps,
                {"nodes": (u["to"],), "orders": (u["id"],)}, done,
                [f"When: simulated time {clock(ds[i]['t'])}–{clock(done)}.",
                 "The question here is different: not how to group boxes, but when to serve an "
                 "urgent order.",
                 f"What happens: plasma #{u['id']} at {name(u['from'])} has the earliest deadline "
                 f"({clock(u['deadline'])}, {left:.0f} min away). The drone first takes a box "
                 f"waiting at {name(o0['from'])} to {name(o0['to'])}, then collects the plasma.",
                 "Watch: whether the plasma still arrives before its deadline."],
                "What this example shows: PPO uses an urgent order's spare time for a short "
                "nearby job, and still delivers it on time."))
    return out


# replay-layout boxes (1280 x 720) a step points at
CHOICES, ACTION = (906, 170, 358, 158), (562, 116, 700, 54)
SLOW = "decisions held for reading · during actions, 1 s of video = 2 simulated minutes"


def stretch_frames(view, steps, until, fps, say):
    """Each decision: its choices spotlighted (fading in), the pick, then its action at 2
    simulated min/s with the whole replay in view."""
    fade = round(0.5 * fps)
    for n, (i, title, line) in enumerate(steps):
        d = view.r["decisions"][i]
        say(title, line)
        for k in range(fade):
            yield view.draw(d["t"], "Deciding", SLOW, ([CHOICES], (k + 1) / fade))
        yield from repeat(view.draw(d["t"], "Deciding", SLOW, ([CHOICES], 1.0)),
                          round(3 * fps))
        yield FADE
        yield from repeat(view.draw(d["t"], "Picked", SLOW, ([ACTION, CHOICES], 1.0)),
                          round(2 * fps))
        m = round(0.6 * fps)
        for k in range(m):
            yield view.draw(d["t"], ("Moving", k / m), SLOW)
        stop = until if n == len(steps) - 1 else d["end"]
        t = d["t"]
        while t < stop - EPS:
            t = min(stop, t + 2 / fps)
            yield view.draw(t, None, SLOW)
    yield from repeat(view.draw(until, None, SLOW), 3 * fps)


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



TITLE = ["One drone delivers medical boxes between a medical center and five hospitals.",
         "Which order should it serve next, and when should it recharge?",
         "We trained a neural network to decide, with Proximal Policy Optimization (PPO),",
         "a reinforcement learning algorithm.", "",
         "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang"]
SLACK = ["What it is: a fixed rule. Unload the boxes for the current place; take waiting boxes "
         "here that go where the cargo already goes; charge if needed; otherwise serve the "
         "earliest deadline first (EDF), with orders of negative slack last.",
         "Slack = time left until the deadline minus the time needed to serve the order now. "
         "Negative slack: the order would be late even if served immediately.",
         "Why not the strongest method? The strongest is the rolling horizon, a planner that "
         "simulates a few steps ahead at every decision. It is about as punctual as PPO but needs "
         "1.4\u00a0s per decision, which a replay cannot show; it is compared in the final test.",
         "Why Slack-aware: at high demand it is the best of the fixed rules that decide in under "
         "1\u00a0ms, as PPO does.",
         "The day: of 40 development days, the one where PPO's lead over Slack-aware was closest "
         "to the median."]
# one line under each method's name on the results slide
DESCRIBE = {"PPO (ours)": "our learned neural network",
            "Rolling horizon": "plans a few steps ahead",
            "Scenario rollout": "tries 16 possible futures",
            "Rollout": "simulates each next action",
            "Insertion": "builds a route by inserting orders",
            "Slack-aware": "EDF with three fixes",
            "Nearest-first": "the job that finishes soonest",
            "EDF": "earliest deadline first",
            "FCFS": "first come, first served",
            "Random": "a random allowed action"}


def speedup(load="high"):
    """How many times faster PPO decides than the rolling horizon."""
    return serial_ms("horizon", "horizon", load) / serial_ms(METHODS[0][3], METHODS[0][4], load)


def summary_lines(rows, ppo, rule):
    p, h = rows[0], rows[1]
    slack = next(row for row in rows if row[0] == "Slack-aware")
    a, b = _lost(ppo, rule)
    place = ppo["map"]["nodes"][a["from"]]["name"]
    return [f"On 30 new maps, PPO delivers {p[2]:.1f}% of orders on time at medium demand and "
            f"{p[3]:.1f}% at high demand; the strongest planner, the rolling horizon, "
            f"{h[2]:.1f}% and {h[3]:.1f}%. At high demand PPO needs {p[4]} per decision, the "
            f"planner {h[4]}.",
            f"At high demand PPO's on-time rate is {p[3] - slack[3]:.1f} percentage points above "
            "Slack-aware's; both decide in under 1\u00a0ms.",
            f"It is not perfect: on the example day it never picked up "
            f"{TYPE_LABELS[a['type']].lower()} #{a['id']} at {place}, which expired; Slack-aware "
            "delivered it on time."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seed", type=int, default=100007)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("outputs/video.mp4"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    fps = args.fps
    ppo, rule = [record_day(name, args.load, args.seed) for name in ("ppo", "slack")]
    r = DuelFrames(ppo, rule)
    view = ReplayFrames(ppo, f"Example day: {args.load} demand on map 0, the map of our earlier "
                        "milestones; this map was not used in training", scale=1.5)
    # each pause zooms into the part it explains (DuelFrames pixel boxes)
    pauses = {t: (title, " ".join(lines), box) for (t, title, lines), box in
              ((ppo_failure(ppo, rule), (20, 90, 940, 529)),  # PPO's map: the expired sample
               (backlog_gap(ppo, rule), (656, 776, 600, 292)),  # the backlog chart
               (day_end(ppo, rule), (1280, 776, 608, 292)))}  # the outcome bars
    end_t = max(pauses)
    lost = _lost(ppo, rule)[0]
    rows = final_rows()
    fast = f"about {round(speedup(), -3):,.0f}× faster"
    task_t = 151.0  # the task's map shows the day at 10:31
    narration = [("Medical center M", "M is the medical center: plasma and medicine are "
                  "collected here. Charger C1 is here too.", 7),
                 ("Five hospitals H1–H5", "Hospitals receive plasma and medicine, send samples "
                  "to M, and send supplies to other hospitals.", 7),
                 ("A second charger C2", "The drone can also charge at C2. Flying to a charger "
                  "and charging both take time.", 7),
                 ("Three no-fly zones", "The hatched areas are no-fly zones; the drone must fly "
                  "around them.", 6),
                 ("Orders and deadlines", "Each order is one box, with a time limit counted from "
                  "its arrival: plasma 30 min, samples 60, supplies 75, medicine 90. Late boxes "
                  "still count, for less; an order expires at twice its limit.", 13),
                 ("One drone", "At most 3 boxes and 2 kg in total. Every flight must leave enough "
                  "battery to reach a charger with at least 15% left.", 10),
                 ("The decision, again and again", "The dispatcher chooses the next pickup, "
                  "delivery or charge (to 60, 80 or 100%). The goal: as many orders on time as "
                  "possible.", 11)]
    sub = Subtitles(*(font_manager.findfont(font_manager.FontProperties(
        family="DejaVu Sans", weight=w)) for w in ("bold", "normal")))
    cue, cues, count, dissolve = ["", "", False], [], [0], [False]

    def say(title, body="", paused=False):
        cue[:] = [title, body, paused]
        cues.append((count[0] / fps, title, body, paused))
        dissolve[0] = True  # a new line of subtitles starts with a dissolve

    section = lambda k, line: (say(f"{k + 1}  {OUTLINE[k]}", line),
                               repeat(r.outline(OUTLINE, k), 6 * fps))[1]

    def frames():
        say("Learning to dispatch a medical delivery drone",
            "Our trained neural network chooses the drone's next pickup, delivery or charging "
            "action.")
        yield from repeat(r.card("Learning to dispatch a medical delivery drone", TITLE,
                                 "ME5418 FINAL PROJECT · GROUP 17"), 12 * fps)
        say("Outline", "The delivery task, how PPO chooses actions, one full day against a fixed "
                       "rule, and ten methods compared on new maps.")
        yield from repeat(r.outline(OUTLINE), 8 * fps)
        yield from section(0, "One drone collects and delivers medical boxes between a medical "
                              "center and five hospitals.")
        for k, (head, line, sec) in enumerate(narration):
            say(head, line)
            yield from repeat(r.task(k, task_t), sec * fps)
        yield from section(1, "PPO uses the drone's state and the waiting orders to choose the "
                              "next action.")
        say("From what it sees to an action",
            "The network gives each allowed action a probability; in this video the drone always "
            "takes the most probable one.")
        yield from repeat(r.method("2  How our PPO agent decides", [
            ("What it sees", "Time, battery, place, boxes on board, travel times, and up to 12 "
                             "waiting orders"),
            ("A neural network", "Attention: the drone's state weighs each waiting order and "
                                 "each box on board"),
            ("A probability for each action", "Actions that break the rules (cargo, battery) "
                                              "are masked out, i.e. not allowed"),
            ("It takes the most probable one", "When that action is done, it decides again")], [
            "How it learned: five networks trained with PPO, 12 million decisions each, on 500 "
            "random maps. Reward: +1 per on-time delivery, +0.1 per late one, \u22120.5 per "
            "expired order, and a small energy penalty.",
            f"In the next scenes each decision is slowed down for viewing. In a timing test the "
            f"network needed {rows[0][4]} per decision (1 ms = one thousandth of a second)."]),
            24 * fps)
        scenes = stretches(ppo)
        for n, (title, intro, steps, focus, until, plan, shows) in enumerate(scenes):
            head = f"Example {n + 1} of {len(scenes)}: {title}"
            say(head, shows)
            yield from repeat(r.bullets(title, plan, f"2  HOW OUR PPO AGENT DECIDES · EXAMPLE "
                                        f"{n + 1} OF {len(scenes)}", shows), 18 * fps)
            say(head, intro)
            view.focus, view.banner = focus, f"Example {n + 1} of {len(scenes)} · {title}"
            first = ppo["decisions"][steps[0][0]]
            yield from repeat(view.draw(first["t"], None, SLOW), 7 * fps)
            yield from stretch_frames(view, steps, until, fps, say)
        view.focus, view.banner = {"nodes": (), "orders": ()}, ""
        yield from section(2, "PPO and a fixed dispatch rule, Slack-aware, receive the same "
                              "orders on the same map.")
        say("Why compare with Slack-aware?",
            "Slack-aware is a fast fixed rule. The strongest method, the rolling horizon, is too "
            "slow to show in a replay; it appears in the final test.")
        yield from repeat(r.bullets("Why compare with Slack-aware?", SLACK, "3  ONE WHOLE DAY",
                                    size=27), 30 * fps)
        steps = (0, 2, 4)  # map, waiting orders, running totals; shown at 10:30
        for n, step in enumerate(steps):
            boxes, title, lines = r.GUIDE[step]
            say(f"How to read the screen · {n + 1}/{len(steps)} · {title}",
                " ".join(line for line in lines if line))
            yield from repeat(r.guide(150.0, step), 9 * fps)
        times = sorted(set(day_times((ppo, rule), fps, r.end)) | set(pauses))
        anim = max(1, round(ANIM_S * fps))
        seen, since, said = [None, None], [anim * 2, anim * 2], None
        for n, t in enumerate(times):
            for k, rec in enumerate((ppo, rule)):
                i = state_at(rec, min(t, rec["meta"]["end"]))["decision"]
                since[k] = 0 if i != seen[k] else since[k] + 1
                seen[k] = i
            phases = tuple("Deciding" if c < anim else "Picked" if c < 2 * anim else None
                           for c in since)
            if t in pauses:
                title, body, box = pauses[t]
                if t == lost["end"]:  # ring the place of the order PPO lost
                    r.views[0].focus = {"nodes": (lost["from"],), "orders": ()}
                frame = r.draw(t, True, phases, "")
                r.views[0].focus = {"nodes": (), "orders": ()}
                say(title, body, True)
                said = None
                badge = ("❚❚  PAUSED", sub.bold)
                m = round(0.8 * fps)
                for k in range(m + 1):  # zoom in, hold, zoom out
                    yield zoom(frame, box, k / m, badge)
                yield from repeat(zoom(frame, box, 1.0, badge),
                                  (PAUSE_S + 3 * (t == end_t)) * fps)
                for k in range(m, -1, -1):
                    yield zoom(frame, box, k / m, badge if k else None)
                continue
            title, lines, _ = phase_caption(ppo["meta"], t)
            if title != said:
                say(title, " ".join(lines))
                said = title
            frame = r.draw(t, False, phases, "playback: 4–12 simulated minutes per second")
            yield frame
            if n % (fps * 20) == 0:
                print(f"Rendering {clock(t)} ({n + 1}/{len(times)})", flush=True)
        yield from section(3, "Ten methods, each tested on 30 new maps never used in training, "
                              "10 days per map: 300 days per demand level.")
        say("Final test: mean on-time rate over 300 days per demand level",
            f"PPO {rows[0][2]:.1f}% vs rolling horizon {rows[1][2]:.1f}% at medium demand, "
            f"{rows[0][3]:.1f}% vs {rows[1][3]:.1f}% at high demand. At high demand PPO decides "
            f"{fast}.")
        yield from repeat(r.results("Final test: 30 new maps × 10 days = 300 days per demand "
                                    "level", [(*row, DESCRIBE[row[0]]) for row in rows],
                                    "On-time rate = share of all orders delivered by their "
                                    "deadline. PPO: mean of five trained networks. Time per "
                                    "decision: separate test, one CPU thread, high demand.",
                                    f"PPO {rows[0][2]:.1f}% vs rolling horizon "
                                    f"{rows[1][2]:.1f}% (medium), {rows[0][3]:.1f}% vs "
                                    f"{rows[1][3]:.1f}% (high); {fast} per decision at high "
                                    "demand."), 30 * fps)
        say("Summary", "PPO is as punctual as the strongest planner at a tiny fraction of its "
                       "decision time, but it still misses some orders.")
        yield from repeat(r.bullets("Summary", summary_lines(rows, ppo, rule), "ME5418 GROUP 17"),
                          25 * fps)
        say("Thank you for watching", "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang")
        yield from repeat(r.card("Thank you for watching", [], "ME5418 GROUP 17"), 4 * fps)

    def with_subtitles(frames):
        """Subtitles above each frame; each new subtitle line, and each FADE marker, dissolves
        from the last frame shown into the next ones over FADE_S, like a slide transition."""
        last, left = None, 0
        for frame in frames:
            if isinstance(frame, str):  # FADE
                dissolve[0] = True
                continue
            count[0] += 1
            out = np.vstack([sub.band(*cue), frame])
            if dissolve[0] and last is not None:
                start, left, dissolve[0] = last, round(FADE_S * fps), False
            if left:
                a = 1 - left / (round(FADE_S * fps) + 1)
                out = (start * (1 - a) + out * a).astype(np.uint8)
                left -= 1
            dissolve[0] = False
            last = out
            yield out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = save_animation(with_subtitles(frames()), str(args.out), fps)
    args.out.with_suffix(".json").write_text(json.dumps(cues, indent=0))  # for translated copies
    print(f"Saved {args.out}: {n} frames, {n / fps:.1f} s")


if __name__ == "__main__":
    main()
