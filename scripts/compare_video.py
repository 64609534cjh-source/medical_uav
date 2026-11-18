"""The group video: PPO and Slack-aware on the same high-load day, short captions throughout.

Title; how to read the side-by-side screen (the task's map, cargo limits and time limits are
introduced here); the whole day side by side with running charts; in the first demand peak the
two maps grow into a slowed close-up of two stretches, each opened by a card on what to watch
and closed by a card on the result; pauses at moments picked from the two records; the final
test's results; a summary.
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
from medical_uav.env.video import (TYPE_LABELS, Band, Captions, DuelFrames, clock, day_times,
                                   flown, morph, zoom)
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
ANIM_S = 0.33  # seconds per step of the decision-card animation
FADE, FADE_S = "fade", 0.4  # marker in a frame stream for a dissolve, and its length (s)
FLASH_S, APPEAR_S = 0.8, 0.35  # a new caption brightens its band this long, and slides in


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


def _waiting(r, t):
    return len(state_at(r, t)["waiting"])


def backlog_gap(ppo, rule):
    """In the last demand peak, the minute the rule's backlog exceeds PPO's the most."""
    a, b = ppo["meta"]["peaks"][-1]
    t = max(np.arange(a, b, 1.0), key=lambda m: _waiting(rule, m) - _waiting(ppo, m))
    return float(t), f"{clock(t)} · Orders waiting"


def phase_caption(meta, t):
    """The label shown while the day plays: which part of the day this is."""
    peaks = meta["peaks"]
    if t < peaks[0][0]:
        return "Morning · few orders"
    if any(a <= t < b for a, b in peaks):
        return "Demand peak · about 3.3× as many orders"
    if t < peaks[-1][0]:
        return "Between the peaks"
    return f"After the second peak · new orders stop at {clock(meta['release_window'])}"


# ---------- two stretches of the first demand peak, shown close up for both ----------
def _picked(d):
    kind, arg = decode(d["action"])
    return d["visible"][arg] if kind == "pick" else None


def _urgent(r, d, orders):
    s = state_at(r, d["t"])
    return min(s["waiting"], key=lambda j: orders[j]["deadline"]) if s["waiting"] else None


def _window(ppo, rule, a, b):
    """On-time deliveries and empty km flown between a and b: (PPO's, the rule's) each."""
    n = [state_at(r, b)["on_time"] - state_at(r, a)["on_time"] for r in (ppo, rule)]
    return n, [flown(r, b, True) - flown(r, a, True) for r in (ppo, rule)]


def stretches(ppo, rule):
    """PPO's two ways of flying less empty, one stretch of its first demand peak each; each a
    dict: name, its one-line description, what each side does (Slack-aware's
    callout, PPO's), moments to pause at (time, PPO's callout), the result callout,
    the orders to highlight, the place to ring, start and end (what is counted) and the
    time playback starts (the second stretch from its first plasma's arrival, so the viewer
    sees PPO finish other work first):
    1. combining jobs: PPO collects three boxes for one hospital at three places and delivers
       them in one visit; the rule only adds boxes waiting where it already is;
    2. using spare time: PPO lets the plasma with the earliest deadline wait for a short nearby
       job, then carries two plasma boxes on one trip; the rule fetches them at once."""
    r = ppo
    a, b = r["meta"]["peaks"][0]
    ds, orders, nodes = r["decisions"], {o["id"]: o for o in r["orders"]}, r["map"]["nodes"]
    name = lambda k: nodes[k]["name"]
    peak = [i for i, d in enumerate(ds) if a <= d["t"] < b]
    result = lambda a, b, n, e: (f"{clock(a)}–{clock(b)} · on time: PPO {n[0]}, Slack-aware {n[1]}"
                                 f"\nEmpty flight: PPO {e[0]:.1f} km, Slack-aware {e[1]:.1f} km")
    out = []

    end = next(i for i in peak if ds[i]["facts"][1].startswith("Unloads 3 of the 3"))
    boxes = sorted((orders[k] for k in state_at(r, ds[end]["t"])["cargo"]),
                   key=lambda o: o["pickup"])
    target = boxes[0]["to"]
    first = min(i for i in range(end) if _picked(ds[i]) in {o["id"] for o in boxes})
    start = max(j for j in range(first + 1) if not state_at(r, ds[j]["t"])["cargo"])
    t0, t1 = ds[start]["t"], ds[end]["end"]
    out.append({"name": "Way 1 · Combining jobs",
                "line": "PPO picks up boxes going the same way",
                "notes": ("Slack-aware: only batches boxes\nwaiting where it already is",
                          f"PPO: collects the three {name(target)} boxes\nat three places"),
                "moments": [(max(g[2] for g in r["segments"] if g[0] == "fly" and g[4] == target
                                 and g[2] <= min(o["end"] for o in boxes) + EPS),  # at the hospital
                             f"All three {name(target)} boxes\narrive in one visit")],
                "result": result(t0, t1, *_window(ppo, rule, t0, t1)),
                "ids": tuple((o["id"], str(j + 1)) for j, o in enumerate(boxes)),
                "place": target, "start": t0, "end": t1, "shown": t0})

    i = next(i for i in peak if i > end and _picked(ds[i]) is not None
             and _urgent(r, ds[i], orders) != _picked(ds[i])
             and orders[_urgent(r, ds[i], orders)]["type"] == 0
             and _outcome(orders[_urgent(r, ds[i], orders)]) == "on time")
    u = orders[_urgent(r, ds[i], orders)]
    pair = sorted((o for o in r["orders"] if o["type"] == 0 and o["to"] == u["to"]
                   and o["pickup"] is not None and ds[i]["t"] <= o["pickup"] and
                   o["end"] <= u["end"] + 5), key=lambda o: o["deadline"])
    t0, t1 = ds[i]["t"], max(o["end"] for o in pair)
    o0 = orders[_picked(ds[i])]  # the short job PPO does first
    out.append({"name": "Way 2 · Using spare time",
                "line": "PPO does a nearby job before an urgent one",
                "notes": (f"At {clock(t0)}, Slack-aware fetches the\ntwo urgent plasmas for "
                          f"{name(u['to'])} at once",
                          f"At {clock(t0)}, PPO ({u['deadline'] - t0:.0f} min left) first does\na "
                          "nearby job, then takes both"),
                "moments": [(o0["end"], f"First the nearby job:\n{TYPE_LABELS[o0['type']].lower()} "
                             f"{name(o0['from'])} → {name(o0['to'])}"),
                            (max(o["pickup"] for o in pair), "Both plasmas on board,\nstill in "
                             "time")],
                "result": result(t0, t1, *_window(ppo, rule, t0, t1)),
                "ids": tuple((o["id"], str(j + 1)) for j, o in enumerate([o0, *pair])),
                "place": u["to"], "start": t0, "end": t1,
                "shown": min(o["release"] for o in pair)})  # played from the plasma's arrival
    return out


CLOSE_RATE = 1.25  # simulated min per second in a close-up


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



VERSUS = [("PPO (ours)", "Neural network trained by reinforcement learning", ""),
          ("Slack-aware", "Hand-written rule: earliest deadline first, plus rules for combining "
                          "boxes and charging", "Best rule under 1 ms")]
TITLE = ["Jihang Cai, Yu Liu, Yufeng Zhang"]
# one line under each method's name on the results slide
DESCRIBE = {"PPO (ours)": "our learned neural network",
            "Rolling horizon": "plans a few steps ahead",
            "Scenario rollout": "tries 16 possible futures",
            "Rollout": "simulates each next action",
            "Insertion": "builds a route by inserting orders",
            "Slack-aware": "EDF with three fixes",
            "Nearest-first": "shortest flight plus handling time",
            "EDF": "earliest deadline first",
            "FCFS": "first come, first served",
            "Random": "a random allowed action"}


def speedup(load="high"):
    """How many times faster PPO decides than the rolling horizon."""
    return serial_ms("horizon", "horizon", load) / serial_ms(METHODS[0][3], METHODS[0][4], load)


def charging(load="medium"):
    """Charges per day and battery (%) entering a demand peak on the held-out maps: PPO's five
    runs, then Slack-aware, as in the report's operations table."""
    acc = {"final": ([], []), "slack": ([], [])}
    with (ROOT / "evidence/final_test/maps.csv").open() as f:
        for row in csv.DictReader(f):
            m = row["method"]
            m = "final" if m.startswith("f12_mm_e05_s") and len(m) == 14 else m  # the five runs
            if m in acc and row["load"] == load:
                acc[m][0].append(float(row["charges"]))
                acc[m][1].append(float(row["battery_at_peak"]))
    return [tuple(float(np.mean(v)) for v in acc[m]) for m in ("final", "slack")]


def summary_lines(rows):
    """The three summary points, from the final test."""
    p, h = rows[0], rows[1]
    slack = next(row for row in rows if row[0] == "Slack-aware")
    (cp, bp), (cs, bs) = charging()
    return [f"Against the rolling horizon: equally punctual at medium load, ahead at high load, "
            f"deciding "
            f"{round(speedup('medium'), -2):,.0f}–{round(speedup('high'), -2):,.0f}× faster "
            f"({p[4]} against {h[4]}).",
            f"Against Slack-aware: also under 1 ms per decision, but {p[2] - slack[2]:.0f}–"
            f"{p[3] - slack[3]:.0f} points more orders on time.",
            f"Charging (medium load): PPO tops up about twice as often ({cp:.0f} against {cs:.0f} times a day) and "
            f"enters peaks with more battery ({bp:.0f} against {bs:.0f} %); this alone does not "
            "explain its lead."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seed", type=int, default=100007)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("outputs/video.mp4"))
    parser.add_argument("--translate", type=Path,
                        help="JSON mapping each English caption to another language (a review "
                             "copy); captions not in it stay English")
    parser.add_argument("--fonts", nargs=3, metavar=("BOLD", "REGULAR", "INDEX"),
                        default=["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "2"],
                        help="caption fonts for --translate (default: Noto Sans CJK SC)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    fps = args.fps
    ppo, rule = [record_day(name, args.load, args.seed) for name in ("ppo", "slack")]
    r = DuelFrames(ppo, rule)
    # each pause: stages (caption title, line, box to zoom into, caption place, seconds held);
    # the view moves from stage to stage, then back out
    t2, title2 = backlog_gap(ppo, rule)
    w2 = [len(state_at(rec, t2)["waiting"]) for rec in (rule, ppo)]
    pauses = {t2: [(title2, f"PPO {w2[1]} · Slack-aware {w2[0]}.", (972, 806, 916, 262),
                    "zoom_top", 5)]}  # the waiting-orders chart
    rows = final_rows()
    # reading the screen: guide step (None = the whole frame), subtitle, line, seconds
    # reading the screen: guide step, title, callout text, seconds
    guide = [(0, "How to read the screen · the map", "M: medical center · H: hospitals · "
              "C: chargers\nHatched: no-fly zones", 5),
             (1, "How to read the screen · each drone", "Current action\nBattery and boxes\n"
              "Waiting orders", 5),
             (2, "How to read the screen · running totals", "Orders on time so far\nOrders "
              "waiting right now", 5)]
    if args.translate:
        words = json.loads(args.translate.read_text(encoding="utf-8"))
        bold, regular, index = args.fonts[0], args.fonts[1], int(args.fonts[2])
    else:
        words, index = {}, 0
        bold, regular = (font_manager.findfont(font_manager.FontProperties(
            family="DejaVu Sans", weight=w)) for w in ("bold", "normal"))
    band = Band(bold, index=index)  # the title band above the frame
    captions = Captions(bold, regular, index)  # the floating explanation in a zoomed pause
    paused_label = words.get("PAUSED", "PAUSED")
    tr = lambda text: words.get(text, text)
    cue, cues, count, dissolve, since = ["", "", False, "slide", None], [], [0], [False], [0]
    where = ["slide"]  # the caption place of the scene being rendered

    def say(title, body="", paused=False, place=None, card=None):
        """Set the title band; body is the text of a zoomed pause's floating box, card =
        (heading, text) for the box at the bottom of the frame."""
        cue[:] = [title, body, paused, place or where[0], card]
        cues.append((count[0] / fps, title, body, paused, cue[3], card))
        dissolve[0] = True  # a new caption starts with a dissolve, slides in and flashes
        since[0] = 0

    def close_ups(windows):
        """A pause on the side-by-side frame, then one stretch per way of flying less empty:
        the agenda slide with that way lit; the maps grow into the close-up; Slack-aware's side,
        then PPO's, is lit with a callout on what it does; both play slowed down with the key
        orders highlighted (pausing at the moment that shows the way); the small charts below
        are lit with the result; the maps shrink back."""
        boxes = lambda f: [f(k) for k in range(2)]
        m = round(0.9 * fps)
        say("First demand peak · a closer look")
        yield from repeat(r.draw(windows[0]["shown"], True), 2 * fps)
        agenda = [(w["name"], w["line"]) for w in windows]
        for n, w in enumerate(windows):
            start, until, shown = w["start"], w["end"], w["shown"]
            show = lambda t, badge="SLOW MOTION", spot=None, note="", at=None: r.close(
                t, start, ((w["place"],), (w["place"],)), w["ids"], badge, until=until,
                spot=spot, note=note, at=at)
            say("Why PPO is more efficient")
            yield from repeat(r.ways(agenda, n), (7 if n == 0 else 3) * fps)  # the agenda
            say(w["name"])
            a, b = r.draw(shown), show(shown, "PAUSED")
            yield from repeat(a, round(0.5 * fps))
            yield from (morph(a, b, boxes(r.duel_map), boxes(r.close_map), k / m)
                        for k in range(m + 1))
            for side, note in ((1, w["notes"][0]), (0, w["notes"][1])):  # the rule, then PPO
                say(w["name"])  # a dissolve into the lit side
                yield from repeat(show(shown, "PAUSED", [r.side_box(side)], note), 6 * fps)
            say(w["name"])
            moments = list(w["moments"])
            for t in np.arange(shown, until, CLOSE_RATE / fps):
                if moments and t >= moments[0][0]:
                    say(w["name"])
                    yield from repeat(show(t, "PAUSED", [r.side_box(0)], moments.pop(0)[1]),
                                      4 * fps)
                    say(w["name"])
                yield show(t)
            say(f"{w['name']} · result")
            yield from repeat(show(until, "PAUSED", list(r.STRIP), w["result"], (960, 470)),
                              7 * fps)
            say("")
            a, b = r.draw(until), show(until, "")
            yield from (morph(a, b, boxes(r.duel_map), boxes(r.close_map), k / m)
                        for k in range(m, -1, -1))
        say("Back to the whole day")
        yield from repeat(r.draw(windows[-1]["end"]), round(1.5 * fps))

    def frames():
        """Every frame of the video in order, with the caption set by say()."""
        say("ME5418 Final Project · Group 17")
        yield from repeat(r.closing("Learning to dispatch a medical delivery drone", TITLE, 62),
                          4 * fps)
        say("One day, two dispatchers")
        yield from repeat(r.versus(VERSUS, "Same map, same orders"), 6 * fps)
        where[0] = "duel"
        for step, head, note, sec in guide:
            say(head)
            frame = r.guide(150.0, step, note)  # 10:30
            yield from repeat(frame, sec * fps)
        say("Replay starts · 08:00")
        yield from repeat(r.draw(0.0), round(1.5 * fps))
        times = sorted(set(day_times((ppo, rule), fps, r.end)) | set(pauses))
        anim = max(1, round(ANIM_S * fps))
        seen, since, said = [None, None], [anim * 2, anim * 2], None
        windows, skip = stretches(ppo, rule), 0.0
        for n, t in enumerate(times):
            if windows and t >= windows[0]["shown"]:  # the slowed close-ups, then back
                skip, said = windows[-1]["end"], None
                yield from close_ups(windows)
                windows = []
            if t < skip:
                continue
            for k, rec in enumerate((ppo, rule)):
                i = state_at(rec, min(t, rec["meta"]["end"]))["decision"]
                since[k] = 0 if i != seen[k] else since[k] + 1
                seen[k] = i
            phases = tuple("Deciding" if c < anim else "Picked" if c < 2 * anim else None
                           for c in since)
            if t in pauses:
                frame = r.draw(t, True, phases)
                said, prev = None, None
                m = round(0.8 * fps)
                for title, body, box, place, hold in pauses[t]:  # zoom or pan to each part
                    say(f"PAUSED · {title}", body, True, place)
                    for k in range(m + 1):
                        yield zoom(frame, box, k / m, None, prev)
                    yield from repeat(zoom(frame, box, 1.0), round(hold * fps))
                    prev = box
                say("")
                for k in range(m, -1, -1):  # back out
                    yield zoom(frame, prev, k / m)
                continue
            title = phase_caption(ppo["meta"], t)
            if title != said:  # only a short label while the day plays
                say(title)
                said = title
            frame = r.draw(t, False, phases)
            yield frame
            if n % (fps * 20) == 0:
                print(f"Rendering {clock(t)} ({n + 1}/{len(times)})", flush=True)
        where[0] = "slide"
        say("This day in numbers")
        yield from repeat(r.day_numbers(), 9 * fps)
        say("Final test · 30 unseen maps × 10 days per demand level")
        yield from repeat(r.results("Final test: 30 new maps × 10 days = 300 days per demand "
                                    "level", [(*row, DESCRIBE[row[0]]) for row in rows],
                                    "",
                                    f"PPO is about as punctual as the rolling horizon, at "
                                    f"{rows[0][4]} instead of {rows[1][4]} per decision."),
                          16 * fps)
        say("Summary")
        yield from repeat(r.bullets("Summary", summary_lines(rows), "ME5418 GROUP 17"),
                          10 * fps)
        say("ME5418 Group 17")
        yield from repeat(r.closing("Thank you for watching", [
            "Learning to dispatch a medical delivery drone"]), 4 * fps)

    def with_captions(frames):
        """Each frame with the current caption drawn in. Each new caption and each FADE marker
        dissolve the picture from the last frame shown into the next ones over FADE_S, like a
        slide transition; the caption itself is drawn on top, sliding in and flashing, so old
        and new captions never overlap."""
        last, left = None, 0
        for frame in frames:
            if isinstance(frame, str):  # FADE
                dissolve[0] = True
                continue
            count[0] += 1
            picture = frame
            if dissolve[0] and last is not None:
                start, left, dissolve[0] = last, round(FADE_S * fps), False
            if left:
                a = 1 - left / (round(FADE_S * fps) + 1)
                picture = (start * (1 - a) + frame * a).astype(np.uint8)
                left -= 1
            dissolve[0] = False
            last = picture
            title, body, paused, place, card = cue
            k = since[0]
            since[0] += 1
            appear, flash = min(1.0, k / (APPEAR_S * fps)), max(0.0, 1 - k / (FLASH_S * fps)) ** 2
            head = tr(title)
            if place.startswith("zoom"):  # the explanation floats beside the zoomed part
                picture = captions.draw(picture, tr(title[9:]), tr(body), True, place, appear,
                                        flash)
                head = paused_label
            if card:  # the box at the bottom of the frame
                picture = captions.draw(picture, tr(card[0]), tr(card[1]), False, "card",
                                        appear, flash)
            yield np.vstack([band.draw(head, appear, flash), Band.shade(picture)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = save_animation(with_captions(frames()), str(args.out), fps)
    args.out.with_suffix(".json").write_text(json.dumps(cues, indent=0))  # for translated copies
    print(f"Saved {args.out}: {n} frames, {n / fps:.1f} s")


if __name__ == "__main__":
    main()
