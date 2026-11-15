"""Frames and videos of a finished day, drawn from its replay record in the replay's style.

ReplayFrames(record).draw(t) returns one 1280 x 720 RGB frame: header with the clock and the
outcome counts, the map, the current decision, battery and cargo, the most urgent waiting
orders and the day's timeline. explained_times() paces a stretch of decisions so each can be
read.
"""
from textwrap import wrap

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

from medical_uav.env.config import EPS
from medical_uav.env.replay import state_at

W, H, DPI = 1280, 720, 80
PAPER, PANEL, INK, MUTED, FAINT = "#f3f6f9", "#ffffff", "#13213c", "#5b6b82", "#93a1b5"
LINE, WASH = "#d9e0e8", "#e9eef5"
ONTIME, LATE, EXPIRED = "#11806a", "#9a6500", "#c8402a"
TYPE_COLORS = ("#c0267a", "#2f6fdb", "#7a4fd0", "#9a6b2f")  # plasma, medicine, sample, supplies
TYPE_LABELS = ("Plasma", "Medicine", "Sample", "Supplies")
FOCUS = "#2f6fdb"
# status colours of the replay: (background, text)
DECIDING, PROGRESS = ("#fff0c2", "#6f4600"), ("#dfe9fd", "#1c49b8")
# the replay at 1x: simulated minutes per second while flying or handling boxes (else 3), and the
# steps of a new decision in seconds: the finished action leaves, choices, the pick, its move up
RATES = {"fly": 1.0, "load": 1.5, "unload": 1.5, "dispose": 1.5}
LEAVE_S, DECIDE_S, PICK_S, MOVE_S = 0.35, 0.7, 0.4, 0.6
PEAK_RATE, OFF_RATE, SLOW_RATE = 3.0, 10.0, 1.0  # comparison video: simulated min/s in and
# outside peaks, and in slow motion before a pause


def pt(px: float) -> float:
    """Font size in points for a size in pixels."""
    return px * 72 / DPI


def rounded(x, y, w, h, radius, **style) -> FancyBboxPatch:
    """A rectangle with rounded corners."""
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", **style)


def clock(t: float, start_hour: int = 8) -> str:
    m = int(start_hour * 60 + t + 1e-9)
    return f"{m // 60:02d}:{m % 60:02d}"


class ReplayFrames:
    """Draws any minute of a recorded day."""

    def __init__(self, record: dict, subtitle: str = ""):
        self.r, self.subtitle = record, subtitle
        self.orders = {o["id"]: o for o in record["orders"]}
        self.fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)
        self.canvas = FigureCanvasAgg(self.fig)

    # ---------- helpers ----------
    def _clock(self, t):
        return clock(t, self.r["meta"]["day_start_hour"])

    def _on_time(self, o):
        return o["end"] - o["release"] <= o["limit"] + EPS

    def _due(self, o, t):
        if o["status"] == "expired" and o["end"] <= t + EPS:
            return f"expired {self._clock(o['end'])}", EXPIRED
        left = o["deadline"] - t
        if left >= 0:
            return f"due in {left:.0f} min", LATE if left < 10 else INK
        return f"late {-left:.0f} min", LATE

    def _panel(self, ax, x, y, w, h):
        ax.add_patch(rounded(x, y, w, h, 10, facecolor=PANEL, edgecolor=LINE, lw=1))

    def _text(self, ax, x, y, s, size=13, color=INK, weight="normal", **kw):
        return ax.text(x, y, s, fontsize=pt(size), color=color, weight=weight, va="top", **kw)

    # ---------- frame ----------
    def draw(self, t: float, phase: str | None = None, speed: str = "") -> np.ndarray:
        """The frame at time t; `phase` overrides the decision card's phase line."""
        r, s = self.r, state_at(self.r, t)
        self.fig.clear()
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        self._header(ax, s, t)
        self._map(s, t)
        self._decision(ax, s, t, phase)
        self._state(ax, s, t)
        self._timeline(ax, t, speed)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()

    def _header(self, ax, s, t):
        meta = self._text(ax, 24, 16, "MEDICAL UAV · ONE SIMULATED DAY", 11, MUTED, "bold")
        meta.set_fontstretch("expanded")
        self._text(ax, 24, 32, self.r["meta"]["policy"], 22, INK, "bold")
        self._text(ax, 24, 62, self.subtitle, 13, MUTED)
        self._text(ax, W - 24, 14, self._clock(t), 38, INK, "bold", ha="right")
        self._text(ax, W - 24, 58, "simulated time", 12, MUTED, ha="right")
        chips = [("on time", s["on_time"], ONTIME), ("late", s["late"], LATE),
                 ("expired", s["expired"], EXPIRED), ("waiting", len(s["waiting"]), MUTED)]
        x = 610
        for label, n, color in chips:
            w = 108
            ax.add_patch(rounded(x, 24, w, 34, 17, facecolor=PANEL, edgecolor=LINE, lw=1))
            ax.add_patch(Circle((x + 18, 41), 7, color=color))
            self._text(ax, x + 32, 30, str(n), 16, INK, "bold")
            self._text(ax, x + 32 + 11 * len(str(n)) + 6, 33, label, 12, MUTED)
            x += w + 8

    def _map(self, s, t, region=(24, 92, 520), scale=1.0):
        size = self.r["map"]["size"]
        x0, y0, side = region
        frame = self.fig.add_axes((x0 / W, 1 - (y0 + side) / H, side / W, side / H))
        frame.set_xlim(-0.15, size + 0.15)
        frame.set_ylim(size + 0.15, -0.15)
        frame.axis("off")
        self._draw_map(frame, s, t)
        if scale != 1:
            for text in frame.texts:
                text.set_fontsize(text.get_fontsize() * scale)
            for artist in [*frame.lines, *frame.patches]:
                artist.set_linewidth(artist.get_linewidth() * scale)

    def _draw_map(self, frame, s, t):
        r, size = self.r, self.r["map"]["size"]
        frame.add_patch(rounded(0, 0, size, size, 0.05, facecolor=PANEL, edgecolor=LINE, lw=1))
        for k in np.arange(0.5, size, 0.5):
            frame.plot([k, k], [0, size], color=WASH, lw=0.8)
            frame.plot([0, size], [k, k], color=WASH, lw=0.8)
        for x, y, w, h in r["map"]["zones"]:
            frame.add_patch(Rectangle((x, y), w, h, facecolor="#eef1f5", edgecolor="#b8c1cd",
                                      hatch="////", lw=0.8))
        nodes = r["map"]["nodes"]
        if s["active"] is not None and r["segments"][s["active"]][0] == "fly":
            seg = r["segments"][s["active"]]
            path = np.array(r["map"]["routes"][f"{seg[3]}-{seg[4]}"])
            lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            cut = s["f"] * lengths[-1]
            done = np.vstack([path[lengths <= cut], s["xy"]])
            left = np.vstack([s["xy"], path[lengths > cut]])
            frame.plot(done[:, 0], done[:, 1], color=INK, lw=2.4, solid_capstyle="round")
            frame.plot(left[:, 0], left[:, 1], color=INK, lw=1.6, ls=(0, (3, 3)), alpha=0.6)
            n = nodes[seg[4]]
            frame.add_patch(Circle((n["x"], n["y"]), 0.17, fill=False, ec=INK, lw=1.6, ls="--"))
        for n in nodes:
            if n["role"] == "center":
                frame.add_patch(Circle((n["x"], n["y"]), 0.17, fill=False, ec=ONTIME, lw=1.5))
                frame.add_patch(FancyBboxPatch((n["x"] - 0.1, n["y"] - 0.1), 0.2, 0.2,
                                               boxstyle="round,pad=0,rounding_size=0.03", fc=INK))
                label = "M · C1"
            elif n["role"] == "charger":
                frame.add_patch(Circle((n["x"], n["y"]), 0.13, fc=PANEL, ec=ONTIME, lw=2))
                bolt = [(0.015, -0.09), (-0.045, 0.01), (0.005, 0.01), (-0.015, 0.09),
                        (0.05, -0.015), (0, -0.015)]
                frame.add_patch(Polygon([(n["x"] + a, n["y"] + b) for a, b in bolt], color=ONTIME))
                label = n["name"]
            else:
                frame.add_patch(Circle((n["x"], n["y"]), 0.09, fc=PANEL, ec=INK, lw=2.2))
                label = n["name"]
            up = n["y"] > 0.45
            frame.text(n["x"], n["y"] - 0.27 if up else n["y"] + 0.37, label, ha="center",
                       va="baseline", fontsize=pt(14), weight="bold", color=INK)
        by_node = {}
        for oid in s["waiting"]:
            o = self.orders[oid]
            by_node.setdefault(o["from"], []).append(o)
        for k, lst in by_node.items():
            n = nodes[k]
            for j, o in enumerate(sorted(lst, key=lambda o: o["deadline"])[:6]):
                corner = (n["x"] + 0.22 + (j % 3) * 0.085, n["y"] - 0.1 + (j // 3) * 0.085)
                frame.add_patch(Rectangle(corner, 0.065, 0.065, fc=TYPE_COLORS[o["type"]],
                                          ec=LATE if t > o["deadline"] else "none", lw=1.2))
            if len(lst) > 6:
                frame.text(n["x"] + 0.22, n["y"] + 0.2, f"+{len(lst) - 6}", fontsize=pt(10),
                           color=MUTED)
        # outcomes of the last few minutes, where they happened
        stacked = {}
        for o in r["orders"]:
            if o["end"] is None or o["end"] > t + EPS or t - o["end"] > 4:
                continue
            if o["status"] == "delivered":
                node = o["to"]
            else:
                node = o["from"] if o["pickup"] is None else None
            if node is None:
                continue
            k = stacked[node] = stacked.get(node, 0) + 1
            good = o["status"] == "delivered" and self._on_time(o)
            delivered = o["status"] == "delivered"
            label = ("✓ on time" if good else "late") if delivered else "✕ expired"
            color = (ONTIME if good else LATE) if o["status"] == "delivered" else EXPIRED
            n = nodes[node]
            y = n["y"] - 0.45 - (k - 1) * 0.18
            alpha = max(0.3, 1 - (t - o["end"]) / 4)
            frame.add_patch(rounded(n["x"] - 0.27, y - 0.08, 0.54, 0.15, 0.07, fc=color,
                                    alpha=alpha))
            frame.text(n["x"], y, label, ha="center", va="center", fontsize=pt(10.5), color="white",
                       weight="bold", alpha=alpha)
        x, y = s["xy"]
        frame.add_patch(Circle((x, y), 0.16, fc=PANEL, alpha=0.9))
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            frame.plot([x, x + dx * 0.075], [y, y + dy * 0.075], color=INK, lw=2)
            frame.add_patch(Circle((x + dx * 0.085, y + dy * 0.085), 0.04, fill=False, ec=INK,
                                   lw=1.4))
        frame.add_patch(Circle((x, y), 0.035, fc=INK))
        cargo = sorted((self.orders[i] for i in s["cargo"]), key=lambda o: o["pickup"])
        for k in range(r["meta"]["max_boxes"]):
            o = cargo[k] if k < len(cargo) else None
            expired = o is not None and o["status"] == "expired" and o["end"] <= t + EPS
            fill = (FAINT if expired else TYPE_COLORS[o["type"]]) if o else "none"
            frame.add_patch(Rectangle((x - 0.1125 + k * 0.08, y + 0.13), 0.075, 0.075, fc=fill,
                                      ec="none" if o else FAINT, lw=0.8))
        if s["active"] is not None and r["segments"][s["active"]][0] == "charge":
            bx, by = x + 0.21, y - 0.17  # beside the drone, clear of the place's name
            frame.add_patch(Polygon([(bx + 0.02, by - 0.09), (bx - 0.04, by + 0.01),
                                     (bx, by + 0.01), (bx - 0.02, by + 0.09),
                                     (bx + 0.05, by - 0.02), (bx + 0.01, by - 0.02)],
                                    color=ONTIME))
        frame.text(size - 0.08, size - 0.1, f"{size:g} × {size:g} km", ha="right", fontsize=pt(10),
                   color=MUTED)
        frame.plot([0.12, 1.12], [size - 0.14, size - 0.14], color=INK, lw=1.6)
        frame.text(0.62, size - 0.2, "1 km", ha="center", fontsize=pt(10.5), color=INK)

    def _decision(self, ax, s, t, phase):
        r = self.r
        x, y, w, h = 560, 92, 696, 230
        self._panel(ax, x, y, w, h)
        decs, end = r["decisions"], r["meta"]["end"]
        i = s["decision"]
        d = decs[i] if i >= 0 else None
        if t >= end - 1e-6:
            sm = r["meta"]["summary"]
            self._text(ax, x + 16, y + 14, f"DAY COMPLETE · {self._clock(end)}", 11, MUTED, "bold")
            share = 100 * sm["on_time"] / sm["orders"]
            self._text(ax, x + 16, y + 34, f"{sm['on_time']} of {sm['orders']} orders delivered on "
                       f"time ({share:.1f}%)", 19, INK, "bold")
            self._text(ax, x + 16, y + 74, f"Delivered {sm['delivered']}  ·  late "
                       f"{sm['delivered'] - sm['on_time']}  ·  expired {sm['expired']}  ·  "
                       f"{sm['km']:.1f} km flown  ·  {len(decs)} decisions", 13, MUTED)
            return
        if d is None:
            self._text(ax, x + 16, y + 14, "START OF THE DAY", 11, MUTED, "bold")
            return
        self._text(ax, x + 16, y + 14, f"DECISION {i + 1} OF {len(decs)} · "
                   f"{self._clock(d['t'])}", 11, MUTED, "bold")
        # phase: None (from t), "Chosen" (a still of the decision), or a step of a new decision:
        # ("Leaving", p), "Deciding", "Picked", ("Moving", p) with p the step's progress
        name, p = phase if isinstance(phase, tuple) else (phase, 0.0)
        if name is None:
            name = "Chosen" if abs(t - d["t"]) < 0.02 else "Running"
        deciding = name != "Running"
        if name in ("Running", "Chosen"):
            self._text(ax, x + 16, y + 32, d["text"], 19, INK, "bold")
        elif name == "Leaving" and i > 0:  # the finished action fades out upwards
            self._text(ax, x + 16, y + 32 - 12 * p, decs[i - 1]["text"], 19, INK, "bold",
                       alpha=1 - p)
        label = "Deciding" if deciding else f"In progress · done at {self._clock(d['end'])}"
        bg, fg = DECIDING if deciding else PROGRESS
        bw = 7.4 * len(label) + 20
        ax.add_patch(FancyBboxPatch((x + 16, y + 62), bw, 22,
                                    boxstyle="round,pad=0,rounding_size=11",
                                    fc=bg, ec="#e9bf55" if deciding else "none"))
        self._text(ax, x + 26, y + 65, label, 12, fg, "bold")
        # progress of the action, beside the status
        frac = 0.0 if deciding else min(1.0, (t - d["t"]) / max(d["end"] - d["t"], EPS))
        px = x + 16 + bw + 12
        ax.add_patch(Rectangle((px, y + 71), x + 340 - px, 4, fc=WASH, ec="none"))
        ax.add_patch(Rectangle((px, y + 71), (x + 340 - px) * frac, 4, fc=FOCUS, ec="none"))
        for k, fact in enumerate(d["facts"]):
            ax.add_patch(Circle((x + 22, y + 102 + 20 * k), 2.5, color=FAINT))
            self._text(ax, x + 32, y + 94 + 20 * k, fact, 13)
        if d["top"]:
            cx = x + 360
            shown = [round(100 * prob, 1) for _, prob in d["top"]]
            digits = 2 if len(set(shown)) < len(shown) else 1  # a near tie needs more digits
            self._text(ax, cx, y + 94, "THE NETWORK'S TOP CHOICES", 11, MUTED, "bold")
            for k, (label, prob) in enumerate(d["top"]):
                yy = y + 112 + 30 * k
                if k == 0 and name in ("Picked", "Moving", "Chosen"):
                    ax.add_patch(Rectangle((cx - 6, yy - 4), x + w - 10 - cx, 28, fc=DECIDING[0],
                                           ec="none"))
                self._text(ax, cx, yy, label, 13, INK, "bold" if k == 0 else "normal")
                self._text(ax, x + w - 16, yy, f"{100 * prob:.{digits}f}%", 13, INK, ha="right")
                ax.add_patch(Rectangle((cx, yy + 18), x + w - 16 - cx, 5, fc=WASH, ec="none"))
                ax.add_patch(Rectangle((cx, yy + 18), (x + w - 16 - cx) * prob, 5,
                                       fc=INK if k == 0 else FAINT, ec="none"))
            self._text(ax, cx, y + h - 31, "Policy probabilities, not chances of success.", 11,
                       MUTED)
            self._text(ax, cx, y + h - 15, "The drone takes the most likely action.", 11, MUTED)
            if name == "Moving":  # the chosen action moves up into the action line
                e = p * p * (3 - 2 * p)
                self._text(ax, cx + (x + 16 - cx) * e, y + 112 - 80 * e, d["top"][0][0],
                           13 + 6.5 * e, INK, "bold",
                           bbox=dict(boxstyle="round,pad=0.2", fc=DECIDING[0], ec="none"))

    def _state(self, ax, s, t):
        r = self.r
        x, y, w, h = 560, 332, 340, 288
        self._panel(ax, x, y, w, h)
        b = s["battery"]
        self._text(ax, x + 16, y + 14, "BATTERY", 11, MUTED, "bold")
        self._text(ax, x + w - 16, y + 10, f"{b:.0f}%", 16, INK, "bold", ha="right")
        ax.add_patch(rounded(x + 16, y + 36, w - 32, 10, 5, fc=WASH, ec="none"))
        low = b < r["meta"]["reserve"] + 10
        ax.add_patch(rounded(x + 16, y + 36, (w - 32) * b / 100, 10, 5,
                             fc=EXPIRED if low else ONTIME, ec="none"))
        rx = x + 16 + (w - 32) * r["meta"]["reserve"] / 100
        ax.plot([rx, rx], [y + 34, y + 48], color=INK, lw=1.5, alpha=0.55)
        cargo = sorted((self.orders[i] for i in s["cargo"]), key=lambda o: o["pickup"])
        kg = sum(o["kg"] for o in cargo)
        self._text(ax, x + 16, y + 60, "CARGO", 11, MUTED, "bold")
        self._text(ax, x + w - 16, y + 60, f"{len(cargo)}/{r['meta']['max_boxes']} boxes · "
                   f"{kg:.2f}/{r['meta']['max_kg']:.2f} kg", 12, MUTED, ha="right")
        for k in range(r["meta"]["max_boxes"]):
            yy = y + 82 + 40 * k
            o = cargo[k] if k < len(cargo) else None
            ax.add_patch(rounded(x + 16, yy, w - 32, 32, 6, fc=PANEL, ec=LINE, lw=1,
                                 ls="-" if o else "--"))
            if o:
                ax.add_patch(Rectangle((x + 28, yy + 11), 10, 10, fc=TYPE_COLORS[o["type"]]))
                nodes = r["map"]["nodes"]
                self._text(ax, x + 46, yy + 8,
                           f"{TYPE_LABELS[o['type']]} → {nodes[o['to']]['name']}", 13)
                due, color = self._due(o, t)
                self._text(ax, x + w - 28, yy + 9, due, 12, color, ha="right")
            else:
                self._text(ax, x + 46, yy + 8, "Empty slot", 13, FAINT)
        # waiting orders
        x2, w2 = 910, 346
        self._panel(ax, x2, y, w2, h)
        waiting = sorted((self.orders[i] for i in s["waiting"]),
                         key=lambda o: (o["deadline"], o["id"]))
        self._text(ax, x2 + 16, y + 14, "WAITING ORDERS", 11, MUTED, "bold")
        self._text(ax, x2 + w2 - 16, y + 14, f"{len(waiting)} waiting · most urgent first", 12,
                   MUTED, ha="right")
        nodes = r["map"]["nodes"]
        for k, o in enumerate(waiting[:6]):
            yy = y + 40 + 36 * k
            ax.add_patch(rounded(x2 + 16, yy, w2 - 32, 30, 6, fc=PAPER, ec="none"))
            ax.add_patch(Rectangle((x2 + 28, yy + 10), 10, 10, fc=TYPE_COLORS[o["type"]]))
            route = f"{nodes[o['from']]['name']} → {nodes[o['to']]['name']}"
            self._text(ax, x2 + 46, yy + 7, f"{TYPE_LABELS[o['type']]} {route}", 13)
            due, color = self._due(o, t)
            self._text(ax, x2 + w2 - 28, yy + 8, due, 12, color, ha="right")
        if len(waiting) > 6:
            self._text(ax, x2 + 20, y + 40 + 36 * 6, f"+{len(waiting) - 6} more waiting", 12, MUTED)

    def _timeline(self, ax, t, speed):
        r = self.r
        meta, end = r["meta"], r["meta"]["end"]
        x, y, w, h = 24, 630, W - 48, 78
        self._panel(ax, x, y, w, h)
        L, R = x + 70, x + w - 16
        X = lambda v: L + (R - L) * v / end
        top, bh = y + 10, 34
        for a, b in meta["peaks"]:
            ax.add_patch(Rectangle((X(a), y + 6), X(b) - X(a), h - 26, fc=WASH, ec="none"))
        by = lambda b: top + bh * (1 - b / 100)
        segs = r["segments"]
        xs = [X(sg[1]) for sg in segs] + [X(segs[-1][2])]
        ys = [by(sg[5]) for sg in segs] + [by(segs[-1][6])]
        ax.fill_between(xs, ys, by(0), color=ONTIME, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=ONTIME, lw=1.2)
        self._text(ax, L - 8, top + 10, "battery", 10, MUTED, ha="right")
        ev = y + 54
        self._text(ax, L - 8, ev - 7, "outcomes", 10, MUTED, ha="right")
        for o in r["orders"]:
            if o["end"] is None:
                continue
            if o["status"] == "delivered":
                color = ONTIME if self._on_time(o) else LATE
                ax.add_patch(Circle((X(o["end"]), ev), 2.3, color=color))
            else:
                ax.plot([X(o["end"]) - 3, X(o["end"]) + 3], [ev - 3, ev + 3], color=EXPIRED, lw=1.3)
                ax.plot([X(o["end"]) - 3, X(o["end"]) + 3], [ev + 3, ev - 3], color=EXPIRED, lw=1.3)
        for hh in range(int(end // 60) + 1):
            self._text(ax, X(hh * 60), y + h - 16, self._clock(hh * 60), 10, MUTED, ha="center")
        ax.plot([X(t), X(t)], [y + 4, y + h - 18], color=INK, lw=1.6)
        if speed:
            self._text(ax, R, y - 18, speed, 11, MUTED, ha="right")


class DuelFrames:
    """Two dispatchers on one clock at 1920 x 1080: per side the map, the current action (with
    the network's top choices for a learned policy), battery, cargo and the most urgent waiting
    orders; below, a caption and three running charts (on time so far, backlog, outcome bars)."""

    W, H = 1920, 1080
    colors = ("#e8590c", "#7048e8")  # the report's PPO and Slack-aware colours
    RULE = "Earliest deadline first · unload where it is · batch the same route · " \
           "hopeless orders last"

    def __init__(self, left, right, titles=("PPO agent", "Slack-aware rule")):
        self.views = (ReplayFrames(left), ReplayFrames(right))
        self.titles, self.r = titles, left
        self.fig = Figure(figsize=(self.W / DPI, self.H / DPI), dpi=DPI, facecolor=PAPER)
        self.canvas = FigureCanvasAgg(self.fig)
        self.end = max(v.r["meta"]["end"] for v in self.views)
        minutes = np.arange(0, self.end + 0.5, 0.5)
        self.series = []  # per side: minutes, on time so far, orders waiting
        for v in self.views:
            states = [state_at(v.r, m) for m in minutes]
            self.series.append((minutes, np.array([s["on_time"] for s in states]),
                                np.array([len(s["waiting"]) for s in states])))

    # ---------- helpers ----------
    def _text(self, ax, x, y, s, size=16, color=INK, weight="normal", **kw):
        return ax.text(x, y, s, fontsize=pt(size), color=color, weight=weight, va="top", **kw)

    def _panel(self, ax, x, y, w, h, **style):
        ax.add_patch(rounded(x, y, w, h, 12, **{"facecolor": PANEL, "edgecolor": LINE, "lw": 1,
                                                **style}))

    def _canvas(self):
        self.fig.clear()
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, self.W), ylim=(self.H, 0))
        ax.axis("off")
        return ax

    def _rgb(self):
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()

    def _box(self, x, y, w, h):
        """Axes over a pixel box, in pixel units with y downwards."""
        return self.fig.add_axes((x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H))

    # ---------- frame ----------
    def draw(self, t: float, caption=None, phases=(None, None), speed=None) -> np.ndarray:
        """The frame at time t; caption = (title, lines, paused); phases: per side None or a step
        of a new decision, "Deciding" (the choices shown) then "Picked" (the top one taken)."""
        ax = self._canvas()
        meta = self.r["meta"]
        self._text(ax, 32, 22, "Same orders, two dispatchers", 34, INK, "bold")
        self._text(ax, 32, 70, f"{meta['load'].capitalize()} load · demand seed {meta['seed']} · "
                   "map 0 · identical order arrivals for both", 17, MUTED)
        self._text(ax, self.W - 32, 14, clock(t, meta["day_start_hour"]), 52, INK, "bold",
                   ha="right")
        peak = any(a <= t < b for a, b in meta["peaks"])
        if speed is None:
            speed = (f"{PEAK_RATE:g} simulated min/s" if peak else
                     f"▶▶ fast-forward · {OFF_RATE:g} simulated min/s")
        self._text(ax, self.W - 32, 76, speed, 15, MUTED, ha="right")
        if peak:
            ax.add_patch(rounded(1450, 30, 200, 40, 20, fc="#fff0c2", ec="#e9bf55"))
            self._text(ax, 1550, 39, "DEMAND PEAK", 17, "#6f4600", "bold", ha="center")
        for k, view in enumerate(self.views):
            self._side(ax, view, state_at(view.r, min(t, view.r["meta"]["end"])), t, k,
                       phases[k])
        self._caption(ax, caption)
        self._charts(t)
        return self._rgb()

    def _side(self, ax, view, s, t, k, phase=None):
        """One dispatcher: name and score, map, decision, battery and cargo, all waiting orders."""
        x, color = 32 + 944 * k, self.colors[k]
        self._text(ax, x, 100, self.titles[k], 28, color, "bold")
        chips = (("on time", s["on_time"], ONTIME), ("late", s["late"], LATE),
                 ("expired", s["expired"], EXPIRED), ("waiting", len(s["waiting"]), MUTED))
        cx = x + 328
        for label, n, shade in chips:
            ax.add_patch(rounded(cx, 100, 138, 38, 19, fc=PANEL, ec=LINE, lw=1))
            ax.add_patch(Circle((cx + 19, 119), 7, color=shade))
            self._text(ax, cx + 34, 105, str(n), 20, INK, "bold")
            self._text(ax, cx + 40 + 14 * len(str(n)), 109, label, 14, MUTED)
            cx += 146
        frame = self._box(x, 150, 400, 400)
        size = view.r["map"]["size"]
        frame.set(xlim=(-0.15, size + 0.15), ylim=(size + 0.15, -0.15))
        frame.axis("off")
        view._draw_map(frame, s, t)
        self._decision(ax, view, s, t, x + 416, color, phase)
        self._onboard(ax, view, s, t, x + 416)
        self._pool(ax, view, s, t, x)

    def _decision(self, ax, view, s, t, x, color, phase=None):
        y, w, h = 150, 496, 252
        self._panel(ax, x, y, w, h, **({"edgecolor": "#e9bf55", "lw": 3} if phase else {}))
        r = view.r
        i = s["decision"]
        d = r["decisions"][i] if i >= 0 else None
        done = t >= r["meta"]["end"] - 1e-6
        if done:
            self._text(ax, x + 18, y + 14, "DAY COMPLETE", 13, MUTED, "bold")
            self._text(ax, x + 18, y + 34, "All orders resolved", 20, INK, "bold")
            return
        if d is None:
            self._text(ax, x + 18, y + 14, "START OF THE DAY", 13, MUTED, "bold")
            return
        head = f"{(phase or 'decision').upper()} {i + 1} OF {len(r['decisions'])} · " \
               f"{view._clock(d['t'])}"
        self._text(ax, x + 18, y + 14, head, 13, "#6f4600" if phase else MUTED, "bold")
        lines = ["Choosing the next action …"] if phase == "Deciding" else wrap(d["text"], 40)[:2]
        for j, line in enumerate(lines):
            self._text(ax, x + 18, y + 34 + 26 * j, line, 20,
                       FAINT if phase == "Deciding" else INK, "bold")
        fy = y + 72 + 26 * (len(lines) - 1)
        for j, fact in enumerate(d["facts"][:3]):
            ax.add_patch(Circle((x + 23, fy + 9 + 20 * j), 2.5, color=FAINT))
            self._text(ax, x + 32, fy + 20 * j, fact, 14)
        yy = y + 160
        if d["top"]:
            self._text(ax, x + 18, yy, "THE NETWORK'S TOP CHOICES", 12, MUTED, "bold")
            for j, (label, prob) in enumerate(d["top"]):
                ry = yy + 20 + 22 * j
                if j == 0 and phase == "Picked":
                    ax.add_patch(Rectangle((x + 10, ry - 3), w - 20, 24, fc=DECIDING[0],
                                           ec="none"))
                self._text(ax, x + 18, ry, label, 14, INK, "bold" if j == 0 else "normal")
                self._text(ax, x + w - 18, ry, f"{100 * prob:.1f}%", 14, INK, ha="right")
                ax.add_patch(Rectangle((x + 18, ry + 17), w - 36, 3, fc=WASH, ec="none"))
                ax.add_patch(Rectangle((x + 18, ry + 17), (w - 36) * prob, 3,
                                       fc=color if j == 0 else FAINT, ec="none"))
        else:
            self._text(ax, x + 18, yy, "HAND-WRITTEN RULE", 12, MUTED, "bold")
            for j, line in enumerate(wrap(self.RULE, 56)):
                self._text(ax, x + 18, yy + 20 + 20 * j, line, 14, INK)

    def _onboard(self, ax, view, s, t, x):
        y, w, h = 412, 496, 138
        self._panel(ax, x, y, w, h)
        r = view.r
        b = s["battery"]
        self._text(ax, x + 18, y + 12, "BATTERY", 12, MUTED, "bold")
        note = f"{b:.0f}%"
        if s["active"] is not None and r["segments"][s["active"]][0] == "charge":
            note = f"charging to {r['segments'][s['active']][6]:.0f}% · " + note
        self._text(ax, x + w - 18, y + 9, note, 15, INK, "bold", ha="right")
        bx, bw = x + 100, w - 100 - (230 if "charging" in note else 70)
        ax.add_patch(rounded(bx, y + 15, bw, 10, 5, fc=WASH, ec="none"))
        ax.add_patch(rounded(bx, y + 15, bw * b / 100, 10, 5,
                             fc=EXPIRED if b < r["meta"]["reserve"] + 10 else ONTIME, ec="none"))
        cargo = sorted((view.orders[o] for o in s["cargo"]), key=lambda o: o["pickup"])
        self._text(ax, x + 18, y + 44, f"CARGO · {len(cargo)}/{r['meta']['max_boxes']} BOXES · "
                   f"{sum(o['kg'] for o in cargo):.2f}/{r['meta']['max_kg']:.2f} KG", 12, MUTED,
                   "bold")
        for j in range(r["meta"]["max_boxes"]):
            yy = y + 64 + 23 * j
            if j < len(cargo):
                self._order(ax, view, cargo[j], t, x + 18, yy, w - 36)
            else:
                self._text(ax, x + 38, yy, "empty slot", 14, FAINT)

    def _pool(self, ax, view, s, t, x):
        """Every waiting order, most urgent first, in three columns."""
        y, w, h, rows = 562, 912, 200, 6
        self._panel(ax, x, y, w, h)
        waiting = sorted((view.orders[o] for o in s["waiting"]),
                         key=lambda o: (o["deadline"], o["id"]))
        self._text(ax, x + 18, y + 12, f"WAITING ORDERS · {len(waiting)} · MOST URGENT FIRST", 12,
                   MUTED, "bold")
        if not waiting:
            self._text(ax, x + 18, y + 40, "None", 14, FAINT)
        cw = (w - 36 - 32) / 3
        shown = waiting if len(waiting) <= 3 * rows else waiting[:3 * rows - 1]
        for j, o in enumerate(shown):
            cx, ry = x + 18 + (j // rows) * (cw + 16), y + 38 + 26 * (j % rows)
            ax.add_patch(rounded(cx - 6, ry - 3, cw + 12, 23, 5, fc=PAPER, ec="none"))
            self._order(ax, view, o, t, cx, ry, cw)
        if len(shown) < len(waiting):
            self._text(ax, x + 18 + 2 * (cw + 16), y + 38 + 26 * (rows - 1),
                       f"+{len(waiting) - len(shown)} more", 14, MUTED, "bold")

    def _order(self, ax, view, o, t, x, y, w):
        nodes = view.r["map"]["nodes"]
        ax.add_patch(Rectangle((x, y + 3), 11, 11, fc=TYPE_COLORS[o["type"]]))
        self._text(ax, x + 18, y, f"#{o['id']} {TYPE_LABELS[o['type']]} "
                   f"{nodes[o['from']]['name']}→{nodes[o['to']]['name']}", 14)
        left = o["deadline"] - t
        due, shade = ((f"{left:.0f} min left", LATE if left < 10 else MUTED) if left >= 0 else
                      (f"late {-left:.0f} min", LATE))
        self._text(ax, x + w, y, due, 14, shade, ha="right")

    def _caption(self, ax, caption):
        x, y, w, h = 32, 772, self.W - 64, 64
        if caption is None:
            return
        title, lines, paused = caption
        self._panel(ax, x, y, w, h, facecolor="#fff0c2" if paused else PANEL,
                    edgecolor="#e9bf55" if paused else LINE, lw=2 if paused else 1)
        self._text(ax, x + 20, y + 8, title, 19, "#6f4600" if paused else INK, "bold")
        self._text(ax, x + 20, y + 36, "   ".join(line for line in lines if line), 16, INK)
        lx = x + w - 20  # colour key of the order squares, right-aligned
        for label, shade in reversed(list(zip(TYPE_LABELS, TYPE_COLORS))):
            lx -= 9 * len(label) + 6
            self._text(ax, lx, y + 12, label, 15, INK)
            lx -= 22
            ax.add_patch(Rectangle((lx, y + 15), 13, 13, fc=shade))
            lx -= 18
        self._text(ax, lx, y + 12, "ORDER TYPES", 12, MUTED, "bold", ha="right")

    def _charts(self, t):
        y, h = 846, 222
        peaks = self.r["meta"]["peaks"]
        ax = self.fig.axes[0]
        boxes = [(32, 600), (656, 600), (1280, 608)]
        titles = [("On-time deliveries so far", "higher is better"),
                  ("Orders waiting (backlog)", "lower is better"),
                  ("Outcomes so far", "same orders, both dispatchers")]
        for (x, w), (title, sub) in zip(boxes, titles):
            self._panel(ax, x, y, w, h)
            self._text(ax, x + 18, y + 10, title, 16, INK, "bold")
            self._text(ax, x + w - 18, y + 13, sub, 13, MUTED, ha="right")
        hours = np.arange(0, self.end + 1, 120)
        for (x, w), col in zip(boxes[:2], (1, 2)):
            c = self._box(x + 50, y + 40, w - 74, h - 70)
            top = max(max(s[col]) for s in self.series) * 1.12 + 1
            for a, b in peaks:
                c.axvspan(a, b, color=WASH, lw=0)
            for k, (m, *vals) in enumerate(self.series):
                v = self.views[k]
                now = state_at(v.r, min(t, v.r["meta"]["end"]))
                now = now["on_time"] if col == 1 else len(now["waiting"])
                xs, ys = np.r_[m[m < t], t], np.r_[vals[col - 1][m < t], now]  # ends at t exactly
                c.plot(xs, ys, color=self.colors[k], lw=2.6, drawstyle="steps-post")
                c.plot(t, now, "o", color=self.colors[k], ms=7)
                c.annotate(f"{now}", (t, now), xytext=(8, 4 if k == 0 else -14),
                           textcoords="offset points", fontsize=pt(16), color=self.colors[k],
                           weight="bold")
            c.axvline(t, color=INK, lw=1.2, alpha=0.5)
            c.set(xlim=(0, self.end), ylim=(0, top))
            c.set_xticks(hours, [clock(v) for v in hours])
            c.tick_params(labelsize=pt(13), colors=MUTED, length=0)
            for side in ("top", "right"):
                c.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                c.spines[side].set_color(LINE)
            c.grid(axis="y", color=WASH, lw=1)
            c.set_facecolor(PANEL)
        x, w = boxes[2]
        b = self._box(x + 30, y + 40, w - 50, h - 70)
        states = [state_at(v.r, min(t, v.r["meta"]["end"])) for v in self.views]
        groups = ("on time", "late", "expired")
        top = max(v.r["meta"]["summary"]["orders"] for v in self.views)
        for k, s in enumerate(states):
            vals = (s["on_time"], s["late"], s["expired"])
            xs = np.arange(3) + (k - 0.5) * 0.38
            b.bar(xs, vals, width=0.36, color=self.colors[k], label=self.titles[k])
            for xx, v in zip(xs, vals):
                b.text(xx, v + top * 0.02, str(v), ha="center", va="bottom", fontsize=pt(16),
                       color=INK, weight="bold")
        b.set_xticks(range(3), groups)
        b.set(ylim=(0, top * 1.05))
        b.set_yticks([])
        b.tick_params(labelsize=pt(15), colors=INK, length=0)
        for side in ("top", "right", "left"):
            b.spines[side].set_visible(False)
        b.spines["bottom"].set_color(LINE)
        b.set_facecolor(PANEL)
        b.legend(loc="upper right", frameon=False, fontsize=pt(15))

    # what each step of the screen guide lights up: pixel boxes and the caption
    GUIDE = [([(40, 158, 388, 388), (984, 158, 388, 388)], "The map",
              ["Drone, medical center M, hospitals H1-H5, chargers; hatched: no-fly zones.",
               "Small squares beside a place are orders waiting there."]),
             ([(442, 144, 508, 412), (1386, 144, 508, 412)], "What each dispatcher does now",
              ["The current action and the situation it was chosen in; PPO also shows its",
               "network's most likely actions. Below: battery and the boxes on board."]),
             ([(26, 556, 924, 212), (970, 556, 924, 212)], "Every order still waiting",
              ["Most urgent first, with the minutes left before each deadline.", ""]),
             ([(352, 94, 592, 50), (1296, 94, 592, 50)], "The score so far",
              ["Orders delivered on time, delivered late, expired, and still waiting.", ""]),
             ([(26, 840, 1868, 234)], "Running totals for both",
              ["On-time deliveries, orders waiting (the backlog) and all outcomes,",
               "updated as the day plays."])]

    def guide(self, t: float, step: int = 0) -> np.ndarray:
        """The frame at t, dimmed except the boxes of one guide step, which the caption names."""
        boxes, title, lines = self.GUIDE[step]
        self.draw(t, (f"How to read the screen · {step + 1}/{len(self.GUIDE)} · {title}", lines,
                      False))
        over = self.fig.add_axes((0, 0, 1, 1), zorder=10)
        over.set(xlim=(0, self.W), ylim=(self.H, 0))
        over.axis("off")
        outer = [(0, 0), (self.W, 0), (self.W, self.H), (0, self.H), (0, 0)]
        verts, codes = list(outer), [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        for x, y, w, h in boxes + [(28, 768, self.W - 56, 72)]:  # holes, opposite winding
            verts += [(x, y), (x, y + h), (x + w, y + h), (x + w, y), (x, y)]
            codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        over.add_patch(PathPatch(Path(verts, codes), fc=PAPER, alpha=0.82, ec="none"))
        for x, y, w, h in boxes:
            over.add_patch(rounded(x, y, w, h, 12, fill=False, ec=self.colors[0], lw=3))
        return self._rgb()

    def results(self, title, rows, note, takeaway="") -> np.ndarray:
        """A bar chart card; rows = (label, colour, medium %, high %, ms per decision)."""
        ax = self._canvas()
        self._panel(ax, 64, 56, self.W - 128, self.H - 112)
        self._text(ax, 110, 90, "RESULTS", 16, MUTED, "bold")
        self._text(ax, 110, 120, title, 38, INK, "bold")
        self._text(ax, 110, 180, note, 19, MUTED)
        for j, load in enumerate(("Medium load", "High load")):
            c = self._box(420 + 560 * j, 270, 500, 620)
            vals = [row[2 + j] for row in rows]
            ys = np.arange(len(rows))[::-1]
            c.barh(ys, vals, color=[row[1] for row in rows], height=0.62)
            for yy, v in zip(ys, vals):
                c.text(v - 1, yy, f"{v:.1f}%", ha="right", va="center", fontsize=pt(18),
                       color="white", weight="bold")
            c.set(xlim=(0, 100), ylim=(-0.6, len(rows) - 0.4))
            c.set_yticks([])
            c.set_xticks([0, 50, 100], ["0", "50", "100 %"])
            c.tick_params(labelsize=pt(14), colors=MUTED, length=0)
            for side in ("top", "right", "left"):
                c.spines[side].set_visible(False)
            c.spines["bottom"].set_color(LINE)
            c.set_title(f"{load} · on time", fontsize=pt(22), color=INK, weight="bold",
                        loc="left", pad=16)
        step = 620 / len(rows)
        for i, row in enumerate(rows):
            yy = 270 + step * (i + 0.5) - 14
            self._text(ax, 110, yy, row[0], 21, INK, "bold" if i == 0 else "normal")
            self._text(ax, 1700, yy, row[4], 21, INK, ha="right")
        self._text(ax, 1700, 240, "TIME PER DECISION", 15, MUTED, "bold", ha="right")
        self._text(ax, 110, 945, takeaway, 24, INK, "bold")
        return self._rgb()

    def card(self, title, lines, kicker) -> np.ndarray:
        """A full-screen text card."""
        ax = self._canvas()
        self._panel(ax, 96, 96, self.W - 192, self.H - 192)
        self._text(ax, 150, 150, kicker, 18, MUTED, "bold")
        self._text(ax, 150, 196, title, 46, INK, "bold")
        for j, line in enumerate(lines):
            self._text(ax, 150, 300 + 50 * j, line, 28)
        return self._rgb()


def slow_times(records, start, end, fps, decide_s=1.2, pick_s=1.2) -> list:
    """(time, phases) from start to end at SLOW_RATE simulated min/s; at each decision of either
    side in the window the clock stops: its choices are shown, then the pick."""
    stops = sorted({d["t"] for r in records for d in r["decisions"] if start < d["t"] <= end})
    out, t = [], start
    for stop in stops + [end]:
        while t < stop - EPS:
            out.append((t, (None, None)))
            t = min(stop, t + SLOW_RATE / fps)
        t = stop
        if stop in stops:
            sides = tuple(any(abs(d["t"] - stop) < EPS for d in r["decisions"]) for r in records)
            for name, sec in (("Deciding", decide_s), ("Picked", pick_s)):
                out += [(stop, tuple(name if k else None for k in sides))] * round(sec * fps)
    return out + [(end, (None, None))]


def duel_times(peaks, end, fps, start=0.0) -> list[float]:
    """Frame times from start to end: PEAK_RATE simulated min/s in demand peaks, else OFF_RATE."""
    marks = sorted({start, end} | {v for w in peaks for v in w if start < v < end})
    times = [start]
    for a, b in zip(marks, marks[1:]):
        step = (PEAK_RATE if any(p <= a < q for p, q in peaks) else OFF_RATE) / fps
        n = int(np.ceil((b - a) / step - EPS))
        times += [a + (b - a) * k / n for k in range(1, n + 1)]
    return times


def explained_times(record: dict, first: int, count: int, fps: int) -> list:
    """(time, phase) for `count` decisions from index `first`, paced like the replay at 1x: the
    steps of each new decision while the clock stands still, then its action."""
    segs, out = record["segments"], []
    for d in record["decisions"][first:first + count]:
        n, m = round(LEAVE_S * fps), round(MOVE_S * fps)
        out += [(d["t"], ("Leaving", k / n)) for k in range(n)]
        out += [(d["t"], "Deciding")] * round(DECIDE_S * fps)
        out += [(d["t"], "Picked")] * round(PICK_S * fps)
        out += [(d["t"], ("Moving", k / m)) for k in range(m)]
        t = d["t"]
        while t < d["end"] - EPS:
            out.append((t, None))
            a = state_at(record, t)["active"]
            t = min(d["end"], t + (RATES.get(segs[a][0], 3.0) if a is not None else 3.0) / fps)
    return out
