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
OUTLINE = ["The task", "How our PPO agent decides", "One whole day: PPO vs the best simple rule",
           "Results on 30 new maps"]
PAUSE_S, ANIM = 7, 4  # pause length (s); frames per step of the decision-card animation
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


def ppo_failure(ppo, rule):
    """The first order PPO lets expire that the rule delivers on time, and why: PPO never went
    there in time, the rule (serving by deadline) did."""
    a, b = sorted(((a, b) for a, b in zip(ppo["orders"], rule["orders"])
                   if _outcome(a) == "expired" and _outcome(b) == "on time"),
                  key=lambda ab: ab[0]["end"])[0]
    place = ppo["map"]["nodes"][a["from"]]["name"]
    where = (f"PPO kept serving orders elsewhere and never flew to {place} in time"
             if a["pickup"] is None else "PPO picked it up but did not deliver it in time")
    return (a["end"], f"{clock(a['end'])} · PPO is not perfect: it loses {_name(ppo, a)}",
            [f"Why: {where}. The rule, which serves strictly by deadline, picked it up at "
             f"{clock(b['pickup'])} and delivered it on time."])


def _waiting(r, t):
    return len(state_at(r, t)["waiting"])


def _per_stop(r):
    """Boxes delivered per delivery stop over the day."""
    stops = [d for d in r["decisions"] if decode(d["action"])[0] == "deliver"]
    return sum(1 for o in r["orders"] if o["status"] == "delivered") / len(stops)


def backlog_gap(ppo, rule):
    """In the last demand peak, the minute the rule's backlog exceeds PPO's the most, and the
    day's flying distance and boxes per stop behind it."""
    a, b = ppo["meta"]["peaks"][-1]
    t = max(np.arange(a, b, 1.0), key=lambda m: _waiting(rule, m) - _waiting(ppo, m))
    sp, sr = state_at(ppo, t), state_at(rule, t)
    km = lambda r: r["meta"]["summary"]["km"]
    return (float(t), f"{clock(t)} · Second peak: {len(sr['waiting'])} orders wait for the rule, "
            f"{len(sp['waiting'])} for PPO",
            [f"Why: PPO groups nearby work. Over the day it delivers {_per_stop(ppo):.1f} boxes "
             f"per stop against {_per_stop(rule):.1f}, and flies {km(ppo):.0f} km against "
             f"{km(rule):.0f} km, which leaves more time for waiting orders."])


def day_end(ppo, rule):
    p, r = ppo["meta"]["summary"], rule["meta"]["summary"]
    t = max(ppo["meta"]["end"], rule["meta"]["end"])
    line = lambda m: (f"{m['on_time']} on time, {m['delivered'] - m['on_time']} late, "
                      f"{m['expired']} expired")
    return (t, f"End of the day · the same {p['orders']} orders",
            [f"PPO: {line(p)}.   The rule: {line(r)}.",
             "One day is one example; the final test below averages 300 days per load."])


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
    """What a delivery unloads, and how many of the boxes in `keep` stay on board."""
    got = _delivered(r, d)
    line = (f"Unloads {len(got)} box{'es' if len(got) > 1 else ''} at "
            f"{nodes[got[0]['to']]['name']}, " +
            ("on time." if len(got) == 1 and _outcome(got[0]) == "on time" else
             "all on time." if all(_outcome(o) == "on time" for o in got) else "not all on time."))
    kept = [k for k in state_at(r, d["end"])["cargo"] if k in keep]
    if kept:
        line += (f" {len(kept)} of the collected boxes stay{'s' if len(kept) == 1 else ''} "
                 "on board.")
    return line


def stretches(r):
    """Two stretches of the first demand peak, each (title, intro, steps, focus, until) with
    steps (decision index, subtitle title, subtitle line):
    1. after a delivery it does not fly back empty, and collects boxes for one hospital at
       three places, delivering them in one stop;
    2. it lets the most urgent order (a plasma) wait for a short errand, then carries two plasma
       boxes on one trip after a short top-up, and the plasma still arrives on time."""
    a, b = r["meta"]["peaks"][0]
    ds, orders, nodes = r["decisions"], {o["id"]: o for o in r["orders"]}, r["map"]["nodes"]
    name = lambda k: nodes[k]["name"]
    peak = [i for i, d in enumerate(ds) if a <= d["t"] < b]
    out = []

    end = next(i for i in peak if ds[i]["facts"][1].startswith("Frees 3 of 3"))
    boxes = [orders[k] for k in state_at(r, ds[end]["t"])["cargo"]]
    target = boxes[0]["to"]
    first = min(i for i in range(end) if _picked(ds[i]) in {o["id"] for o in boxes})
    start = max(j for j in range(first + 1) if not state_at(r, ds[j]["t"])["cargo"])
    steps, n = [], 0
    for i in range(start, end + 1):
        d = ds[i]
        k = _picked(d)
        if k is not None:
            o, u = orders[k], _urgent(r, d, orders)
            line = f"{TYPE_LABELS[o['type']]} #{k} for {name(o['to'])}, due {clock(o['deadline'])}."
            if i == start:
                line = (f"It has just delivered here; instead of flying back empty it takes "
                        f"{_label(o, nodes)}.")
            if o["to"] == target:
                n += 1
                line += f" Box {n} of 3 for {name(target)}."
            if u is not None and u != k:
                line += (f" Not the most urgent: {_label(orders[u], nodes)} is due "
                         f"{clock(orders[u]['deadline'])}.")
        elif i == end:
            places = ", ".join(name(o["from"]) for o in sorted(boxes, key=lambda o: o["pickup"]))
            line = (f"All three boxes for {name(target)}, picked up at {places}, arrive in one "
                    "stop" + (", all on time." if all(_outcome(o) == "on time" for o in boxes)
                              else "."))
        else:
            line = _deliver_line(r, d, nodes, keep={o["id"] for o in boxes})
        steps.append((i, f"{clock(d['t'])} · {d['text']}", line))
    out.append((f"No empty trips, and boxes for {name(target)} collected on the way",
                f"Watch the boxes for {name(target)} (highlighted): picked up at three places, "
                "delivered together.", steps,
                {"nodes": (target,), "orders": tuple(o["id"] for o in boxes)}, ds[end]["end"],
                [f"When: {clock(ds[start]['t'])} to {clock(ds[end]['end'])}, in the first demand "
                 "peak.",
                 "What happens: after a delivery the drone does not fly back empty, and it "
                 f"collects three boxes for {name(target)} at three different places.",
                 f"Watch: the highlighted boxes for {name(target)}, and the order in which the "
                 "boxes are picked up."],
                "What PPO learned: plan trips, not single orders. No empty flights, and boxes "
                "for the same place collected along the way."))

    i = next(i for i in peak if i > end and _picked(ds[i]) is not None
             and _urgent(r, ds[i], orders) != _picked(ds[i])
             and orders[_urgent(r, ds[i], orders)]["type"] == 0
             and _outcome(orders[_urgent(r, ds[i], orders)]) == "on time")
    u = orders[_urgent(r, ds[i], orders)]
    last = next(j for j in range(i, len(ds)) if u in _delivered(r, ds[j]))
    steps = []
    for j in range(i, last + 1):
        d = ds[j]
        k = _picked(d)
        kind = decode(d["action"])[0]
        if j == i:
            o = orders[k]
            back = next(dd["t"] for dd in ds[i + 1:] if _picked(dd) == u["id"])
            line = (f"The most urgent order is plasma #{u['id']} at {name(u['from'])}, due "
                    f"{clock(u['deadline'])}, {u['deadline'] - d['t']:.0f} min away. First it "
                    f"takes {_label(o, nodes)}, waiting right here; the errand takes "
                    f"{back - d['t']:.0f} min.")
        elif k is not None:
            o = orders[k]
            mates = [c for c in state_at(r, d["t"])["cargo"] if orders[c]["to"] == o["to"]]
            line = f"{TYPE_LABELS[o['type']]} #{k} for {name(o['to'])}, due {clock(o['deadline'])}."
            if mates:
                line += f" Same destination as #{mates[0]} on board: one trip for both."
        elif kind == "charge":
            seg = next(g for g in r["segments"] if g[0] == "charge" and g[1] >= d["t"] - EPS)
            rate = (seg[6] - seg[5]) / (seg[2] - seg[1])  # battery % per minute
            line = (f"Battery {seg[5]:.0f}%: a top-up to {seg[6]:.0f}% takes "
                    f"{seg[2] - seg[1]:.0f} min; to 100% it would take about "
                    f"{(100 - seg[5]) / rate:.0f} min.")
        elif j == last:
            got = _delivered(r, d)
            line = ("Arrivals: " + ", ".join(
                f"#{o['id']} at {clock(o['end'])} (due {clock(o['deadline'])})" for o in got) +
                (": all on time." if all(_outcome(o) == "on time" for o in got) else "."))
        else:
            line = _deliver_line(r, d, nodes)
        steps.append((j, f"{clock(d['t'])} · {d['text']}", line))
    out.append(("The most urgent order can wait a little",
                f"At {clock(ds[i]['t'])} plasma #{u['id']} is the most urgent order. Watch what "
                "PPO does first, and whether the plasma still arrives on time.", steps,
                {"nodes": (u["to"],), "orders": (u["id"],)},
                max(o["end"] for o in _delivered(r, ds[last])),
                [f"When: {clock(ds[i]['t'])} to "
                 f"{clock(max(o['end'] for o in _delivered(r, ds[last])))}.",
                 "Different from scenario 1: the question is not how to group boxes, but when "
                 "to serve an urgent order.",
                 f"What happens: plasma #{u['id']} is the most urgent order, yet it still has "
                 f"{u['deadline'] - ds[i]['t']:.0f} minutes; the drone first does a short job "
                 "where it already is, then fetches the plasma.",
                 "Watch: whether the plasma still arrives on time."],
                "What PPO learned: use the spare time of an urgent order for nearby work, and "
                "still deliver it on time."))
    return out


# replay-layout boxes (1280 x 720) a step points at
CHOICES, ACTION = (906, 170, 358, 158), (562, 116, 700, 54)
SLOW = "decisions slowed down for viewing · flights at 2 simulated min/s"


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
                          round(1.5 * fps))
        yield FADE
        yield from repeat(view.draw(d["t"], "Picked", SLOW, ([ACTION, CHOICES], 1.0)), fps)
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



TITLE = ["One drone, a medical center and five hospitals: which order next, and when to",
         "charge? We trained a neural network with reinforcement learning (PPO) to decide.", "",
         "ME5418 Group 17 · Jihang Cai, Yu Liu, Yufeng Zhang"]
SLACK = ["What it is: the classic earliest-deadline-first rule (EDF), plus three fixes we added: "
         "unload where the drone is, take waiting orders going the same way, and leave orders "
         "that can no longer be on time to last.",
         "Why this one: it is the strongest of the simple rules, and like PPO it decides in under "
         "a millisecond, so both dispatchers work at the same speed.",
         "Planners that search ahead are slower; they are compared in the final test at the end.",
         "The day was fixed in advance: of 40 development days, the one with the median PPO lead "
         "over Slack-aware."]


def speedup(load="high"):
    """How many times faster PPO decides than the rolling horizon."""
    return serial_ms("horizon", "horizon", load) / serial_ms(METHODS[0][3], METHODS[0][4], load)


def summary_lines(rows, ppo, rule):
    p, h = rows[0], rows[1]
    slack = next(row for row in rows if row[0] == "Slack-aware")
    o = sorted((a for a, b in zip(ppo["orders"], rule["orders"])
                if _outcome(a) == "expired" and _outcome(b) == "on time"), key=lambda o: o["end"])[0]
    nodes = ppo["map"]["nodes"]
    lost = ("never went to " + nodes[o["from"]]["name"] + " for" if o["pickup"] is None else
            "did not deliver")
    return [f"On-time rates similar to the strongest planner (rolling horizon: {h[2]:.1f}% vs "
            f"PPO {p[2]:.1f}% at medium, {h[3]:.1f}% vs {p[3]:.1f}% at high load), with "
            f"decisions about {round(speedup(), -3):,.0f}× faster ({p[4]} vs {h[4]}).",
            f"At high load, PPO's on-time rate is {p[3] - slack[3]:.1f} points above the best "
            "simple rule (Slack-aware).",
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
    # each pause zooms into the part it explains (DuelFrames pixel boxes)
    pauses = {t: (title, " ".join(lines), box) for (t, title, lines), box in
              ((ppo_failure(ppo, rule), (20, 90, 940, 529)),  # PPO's map: the expired sample
               (backlog_gap(ppo, rule), (656, 776, 600, 292)),  # the backlog chart
               (day_end(ppo, rule), (1280, 776, 608, 292)))}  # the outcome bars
    end_t = max(pauses)
    lost = next(a for a, b in zip(ppo["orders"], rule["orders"])
                if a["end"] == ppo_failure(ppo, rule)[0] and _outcome(a) == "expired")
    rows = final_rows()
    fast = f"about {round(speedup(), -3):,.0f}× faster"
    task_t = 151.0  # the task's map shows the day at 10:31
    narration = [("Medical center M", "Plasma and medicine start here; the charger C1 is here."),
                 ("Five hospitals H1-H5", "They receive boxes, send samples to M and supplies to "
                                          "each other."),
                 ("A second charger C2", "Charging takes time, so where and when to charge "
                                         "matters."),
                 ("Three no-fly zones", "Routes must go around them."),
                 ("Orders arrive all day, each with a deadline",
                  "Plasma within 30 min, samples 60, supplies 75, medicine 90. Late deliveries "
                  "still count for less; after twice its limit an order expires."),
                 ("One drone", "At most 3 boxes and 2 kg on board; the battery must stay above "
                               "15%."),
                 ("The decision, again and again", "Which order to pick up or deliver next, and "
                  "when to charge. The goal: as many orders on time as possible.")]
    sub = Subtitles(*(font_manager.findfont(font_manager.FontProperties(
        family="DejaVu Sans", weight=w)) for w in ("bold", "normal")))
    cue, cues, count, dissolve = ["", "", False], [], [0], [False]

    def say(title, body="", paused=False):
        cue[:] = [title, body, paused]
        cues.append((count[0] / fps, title, body, paused))
        dissolve[0] = True  # a new line of subtitles starts with a dissolve

    section = lambda k, line: (say(f"{k + 1}  {OUTLINE[k]}", line),
                               repeat(r.outline(OUTLINE, k), 3 * fps))[1]

    def frames():
        say("Learning to dispatch a medical delivery drone",
            "ME5418 Group 17 final project: a neural network, trained by reinforcement learning, "
            "dispatches one delivery drone.")
        yield from repeat(r.card("Learning to dispatch a medical delivery drone", TITLE,
                                 "ME5418 FINAL PROJECT · GROUP 17"), 6 * fps)
        say("Outline", "The task, how our agent decides, one whole day against the best "
                       "simple rule, and the final test.")
        yield from repeat(r.outline(OUTLINE), 5 * fps)
        yield from section(0, "First, the world the dispatcher works in.")
        for k, (head, line) in enumerate(narration):
            say(head, line)
            yield from repeat(r.task(k, task_t), (5 if k >= 4 else 3) * fps)
        yield from section(1, "What the network sees, and how it picks an action.")
        say("From what it sees to an action",
            "The network gives every allowed action a probability and takes the most likely one. "
            "It learned this with PPO, from rewards for on-time deliveries.")
        yield from repeat(r.method("2  How our PPO agent decides", [
            ("What it sees", "The drone (battery, place, cargo) and up to 12 waiting orders"),
            ("A neural network", "Attention compares every order with the others and with the "
                                 "drone"),
            ("A probability for each action", "Actions that are impossible now are masked out"),
            ("It takes the most likely one", "Then the next decision, again and again")], [
            "How it learned: PPO over 12 million decisions on 500 random maps. Reward: +1 for "
            "each on-time delivery, less for a late one, a penalty when an order expires.",
            f"In the next scenes each decision is slowed down for viewing; the network itself "
            f"needs {rows[0][4]}."]), 12 * fps)
        scenes = stretches(ppo)
        for n, (title, intro, steps, focus, until, plan, trait) in enumerate(scenes):
            say(f"Scenario {n + 1} of {len(scenes)}: {title}", trait)
            yield from repeat(r.bullets(title, plan, f"2  HOW OUR PPO AGENT DECIDES · SCENARIO "
                                        f"{n + 1} OF {len(scenes)}", trait), 9 * fps)
            say(f"Scenario {n + 1} of {len(scenes)}: {title}", intro)
            view.focus, view.banner = focus, f"Scenario {n + 1} of {len(scenes)} · {title}"
            first = ppo["decisions"][steps[0][0]]
            yield from repeat(view.draw(first["t"], None, SLOW), 3 * fps)
            yield from stretch_frames(view, steps, until, fps, say)
        view.focus, view.banner = {"nodes": (), "orders": ()}, ""
        yield from section(2, "The same day, side by side with the best simple rule.")
        say("The comparison: the Slack-aware rule",
            "Both dispatchers get the same 66 orders on the same map, on one shared clock.")
        yield from repeat(r.bullets("The comparison: the Slack-aware rule", SLACK,
                                    "3  ONE WHOLE DAY"), 12 * fps)
        steps = (0, 2, 4)  # map, waiting orders, running totals; shown at 10:30
        for n, step in enumerate(steps):
            boxes, title, lines = r.GUIDE[step]
            say(f"How to read the screen · {n + 1}/{len(steps)} · {title}",
                " ".join(line for line in lines if line) + " (Shown at 10:30.)")
            yield from repeat(r.guide(150.0, step), 5 * fps)
        times = sorted(set(day_times((ppo, rule), fps, r.end)) | set(pauses))
        seen, since, said = [None, None], [ANIM * 2, ANIM * 2], None
        for n, t in enumerate(times):
            for k, rec in enumerate((ppo, rule)):
                i = state_at(rec, min(t, rec["meta"]["end"]))["decision"]
                since[k] = 0 if i != seen[k] else since[k] + 1
                seen[k] = i
            phases = tuple("Deciding" if c < ANIM else "Picked" if c < 2 * ANIM else None
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
                                  (PAUSE_S + 2 * (t == end_t)) * fps)
                for k in range(m, -1, -1):
                    yield zoom(frame, box, k / m, badge if k else None)
                continue
            title, lines, _ = phase_caption(ppo["meta"], t)
            if title != said:
                say(title, " ".join(lines))
                said = title
            frame = r.draw(t, False, phases, "time-lapse · 4-12 simulated min per second")
            yield frame
            if n % (fps * 20) == 0:
                print(f"Rendering {clock(t)} ({n + 1}/{len(times)})", flush=True)
        yield from section(3, "Six methods, 30 new maps never used in training, ten days each.")
        say("Final test: mean on-time rate over 300 days per load",
            f"PPO {rows[0][2]:.1f}% vs rolling horizon {rows[1][2]:.1f}% at medium load, "
            f"{rows[0][3]:.1f}% vs {rows[1][3]:.1f}% at high load; {fast} per decision.")
        yield from repeat(r.results("Final test: 30 new maps never used in training, 300 days "
                                    "per load", rows, "Mean on-time rate; PPO: mean of five "
                                    "training runs. Time per decision: one CPU thread, high load.",
                                    f"PPO {rows[0][2]:.1f}% vs rolling horizon {rows[1][2]:.1f}% "
                                    f"(medium), {rows[0][3]:.1f}% vs {rows[1][3]:.1f}% (high); "
                                    f"{fast} per decision."), 18 * fps)
        say("Summary", "Thank you for watching. ME5418 Group 17.")
        yield from repeat(r.bullets("Summary", summary_lines(rows, ppo, rule), "ME5418 GROUP 17"),
                          12 * fps)

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
