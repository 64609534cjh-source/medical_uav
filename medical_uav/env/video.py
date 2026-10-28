"""Frames and videos of a finished day, drawn from its replay record in the replay's style.

ReplayFrames(record).draw(t) returns one 1280 x 720 RGB frame: header with the clock and the
outcome counts, the map, the current decision, battery and cargo, the most urgent waiting
orders and the day's timeline. explained_times() paces a stretch of decisions so each can be
read.
"""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

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
FLY_RATE, GROUND_RATE = 1.0, 8.0  # the side-by-side comparison video: flying, everything else


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


class ComparisonFrames(ReplayFrames):
    """Two dispatchers on one clock; notes are a title and up to three short lines."""

    colors = ("#2f6fdb", "#7a4fd0")

    def __init__(self, left, right, titles=("PPO agent", "Slack-aware rule")):
        super().__init__(left)
        self.views = (self, ReplayFrames(right))
        self.views[1].fig = self.fig
        self.titles = titles

    def _blank(self):
        self.fig.clear()
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        return ax

    def _rgb(self):
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()

    def draw(self, t: float, note=None) -> np.ndarray:
        ax = self._blank()
        self._text(ax, 24, 18, "Same orders, two dispatchers", 28, INK, "bold")
        meta = self.r["meta"]
        self._text(ax, 24, 55, f"{meta['load'].capitalize()} load · demand seed {meta['seed']}"
                   " · same arrivals and map", 14, MUTED)
        self._text(ax, 1256, 12, self._clock(t), 36, INK, "bold", ha="right")
        self._text(ax, 1256, 54, "shared simulated time", 12, MUTED, ha="right")
        if any(a <= t < b for a, b in meta["peaks"]):
            ax.add_patch(rounded(850, 27, 140, 28, 14, fc=WASH, ec="none"))
            self._text(ax, 920, 33, "DEMAND PEAK", 12, INK, "bold", ha="center")
        for k, view in enumerate(self.views):
            self._column(ax, view, state_at(view.r, t), t, k)
        self._text(ax, 28, 594, "Flight: 1 simulated min/s · both grounded: 8 min/s", 10, MUTED)
        self._text(ax, 1252, 594, "Outcomes: green on time · amber late · red expired", 10,
                   MUTED, ha="right")
        self._shared_timeline(ax, t)
        if note is not None:
            overlay = self.fig.add_axes((0, 0, 1, 1), zorder=10)
            overlay.set(xlim=(0, W), ylim=(H, 0))
            overlay.axis("off")
            overlay.add_patch(Rectangle((0, 588), W, H - 588, fc=PAPER, ec="none"))
            self._panel(overlay, 24, 590, 1232, 116)
            self._text(overlay, 44, 600, note[0], 17, INK, "bold")
            for j, line in enumerate(note[1:4]):
                self._text(overlay, 44, 628 + 22 * j, line, 14)
        return self._rgb()

    def _column(self, ax, view, s, t, k):
        x, color = 24 + 628 * k, self.colors[k]
        self._text(ax, x, 88, self.titles[k], 22, color, "bold")
        counts = (("on time", s["on_time"], ONTIME), ("late", s["late"], LATE),
                  ("expired", s["expired"], EXPIRED), ("waiting", len(s["waiting"]), MUTED))
        for j, (label, n, shade) in enumerate(counts):
            self._text(ax, x + j * 146, 119, f"{n} {label}", 14, shade, "bold")
        view._map(s, t, (x, 140, 386), scale=0.9)
        self._orders(ax, view, s, t, x + 400)
        d = view.r["decisions"][s["decision"]] if s["decision"] >= 0 else None
        action = d["text"] if d else "Waiting for the day to start"
        if t >= view.r["meta"]["end"]:
            action = "Day complete"
        self._text(ax, x + 4, 536, action, 16, INK, "bold")
        b = s["battery"]
        ax.add_patch(rounded(x + 4, 570, 230, 12, 6, fc=WASH, ec="none"))
        ax.add_patch(rounded(x + 4, 570, 230 * b / 100, 12, 6, fc=color, ec="none"))
        label = f"{b:.0f}%"
        if s["active"] is not None:
            seg = view.r["segments"][s["active"]]
            if seg[0] == "charge":
                label += f" · charging to {seg[6]:.0f}%"
        self._text(ax, x + 248, 565, label, 15, color, "bold")

    def _orders(self, ax, view, s, t, x):
        d = view.r["decisions"][s["decision"]] if s["decision"] >= 0 else None
        visible = d.get("visible", s["waiting"]) if d else []
        orders = [view.orders[i] for i in visible if i in s["waiting"]]
        self._text(ax, x, 150, "VISIBLE ORDERS", 12, MUTED, "bold")
        label = f"At decision {view._clock(d['t'])}" if d else "No decision yet"
        self._text(ax, x, 170, label, 12, MUTED)
        for j, o in enumerate(orders[:4]):
            self._order_line(ax, view, o, t, x, 196 + 43 * j)
        if not orders:
            self._text(ax, x, 198, "None waiting in view", 12, FAINT)
        if len(orders) > 4:
            self._text(ax, x, 368, f"+{len(orders) - 4} more visible", 11, MUTED)
        self._text(ax, x, 392, f"CARGO · {len(s['cargo'])}/{view.r['meta']['max_boxes']}",
                   12, MUTED, "bold")
        for j, oid in enumerate(s["cargo"]):
            self._order_line(ax, view, view.orders[oid], t, x, 414 + 35 * j)

    def _order_line(self, ax, view, o, t, x, y):
        nodes = view.r["map"]["nodes"]
        route = f"{nodes[o['from']]['name']} → {nodes[o['to']]['name']}"
        self._text(ax, x, y, f"{TYPE_LABELS[o['type']]} · {route}", 12,
                   TYPE_COLORS[o["type"]], "bold")
        due, color = view._due(o, t)
        self._text(ax, x, y + 16, f"#{o['id']} · {due}", 11, color)

    def _shared_timeline(self, ax, t):
        self._panel(ax, 24, 610, 1232, 98)
        end = max(view.r["meta"]["end"] for view in self.views)
        X = lambda v: 202 + 1030 * v / end
        for a, b in self.r["meta"]["peaks"]:
            ax.add_patch(Rectangle((X(a), 616), X(b) - X(a), 71, fc=WASH, ec="none"))
        for k, view in enumerate(self.views):
            color, segs = self.colors[k], view.r["segments"]
            xs = [X(sg[1]) for sg in segs] + [X(segs[-1][2])]
            bs = [sg[5] for sg in segs] + [segs[-1][6]]
            ax.plot(xs, [646 - b * 0.27 for b in bs], color=color, lw=1.3)
            self._text(ax, 36, 617 + k * 17, f"{self.titles[k]} battery", 11, color)
            y = 660 + k * 16
            self._text(ax, 36, y - 7, f"{self.titles[k]} outcomes", 11, color)
            for o in view.r["orders"]:
                if o["end"] is None:
                    continue
                shade = EXPIRED if o["status"] == "expired" else (
                    ONTIME if self._on_time(o) else LATE)
                ax.add_patch(Circle((X(o["end"]), y), 2.3, color=shade))
        for minute in range(0, int(end) + 1, 60):
            self._text(ax, X(minute), 691, self._clock(minute), 10, MUTED, ha="center")
        ax.plot([X(t), X(t)], [616, 687], color=INK, lw=1.6)

    def card(self, title, lines, kicker="SAME ORDERS, TWO DISPATCHERS") -> np.ndarray:
        """A full-screen intro or results card in the replay palette."""
        ax = self._blank()
        self._panel(ax, 64, 64, 1152, 592)
        self._text(ax, 100, 96, kicker, 14, MUTED, "bold")
        self._text(ax, 100, 136, title, 32, INK, "bold")
        for j, line in enumerate(lines):
            self._text(ax, 100, 210 + 36 * j, line, 21)
        return self._rgb()


def compare_times(left, right, fps, start, end) -> list[float]:
    """Shared smart speed, splitting at segment boundaries to catch either drone's flights."""
    records = (left, right)
    boundaries = sorted({start, end} | {s[i] for r in records for s in r["segments"]
                                         for i in (1, 2) if start < s[i] < end})
    times = [start]
    for a, b in zip(boundaries, boundaries[1:]):
        flying = any(s[0] == "fly" and s[1] <= (a + b) / 2 < s[2]
                     for r in records for s in r["segments"])
        step = (FLY_RATE if flying else GROUND_RATE) / fps
        t = a
        while t + step < b - EPS:
            t += step
            times.append(t)
        times.append(b)
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
