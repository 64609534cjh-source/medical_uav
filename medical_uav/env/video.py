"""Frames and videos of a finished day, drawn from its replay record in the replay's style.

ReplayFrames(record).draw(t) returns one 1280 x 720 RGB frame: header with the clock and the
outcome counts, the map, the current decision, battery and cargo, the most urgent waiting
orders and the day's timeline. smart_times() paces a whole day like the replay's smart speed;
explained_times() holds a few decisions so each can be read.
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
FLY_RATE, GROUND_RATE, HOLD_S = 1.0, 8.0, 0.6  # smart speed, as in the replay


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

    def _map(self, s, t):
        r, size = self.r, self.r["map"]["size"]
        x0, y0, side = 24, 92, 520
        frame = self.fig.add_axes((x0 / W, 1 - (y0 + side) / H, side / W, side / H))
        frame.set_xlim(-0.15, size + 0.15)
        frame.set_ylim(size + 0.15, -0.15)
        frame.axis("off")
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
        d = decs[s["decision"]] if s["decision"] >= 0 else None
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
        self._text(ax, x + 16, y + 14, f"DECISION {s['decision'] + 1} OF {len(decs)} · "
                   f"{self._clock(d['t'])}", 11, MUTED, "bold")
        self._text(ax, x + 16, y + 32, d["text"], 19, INK, "bold")
        if phase is None and abs(t - d["t"]) < 0.02:
            phase = "Deciding"
        elif phase is None:
            running = t < d["end"] - EPS
            phase = f"In progress · done at {self._clock(d['end'])}" if running else "Done"
        ax.add_patch(FancyBboxPatch((x + 16, y + 62), 7.4 * len(phase) + 20, 22,
                                    boxstyle="round,pad=0,rounding_size=11", fc=WASH, ec="none"))
        self._text(ax, x + 26, y + 65, phase, 12, INK, "bold")
        for k, fact in enumerate(d["facts"]):
            ax.add_patch(Circle((x + 22, y + 102 + 20 * k), 2.5, color=FAINT))
            self._text(ax, x + 32, y + 94 + 20 * k, fact, 13)
        if d["top"]:
            cx = x + 360
            self._text(ax, cx, y + 94, "THE NETWORK'S TOP CHOICES", 11, MUTED, "bold")
            for k, (label, p) in enumerate(d["top"]):
                yy = y + 116 + 34 * k
                self._text(ax, cx, yy, label, 13, INK, "bold" if k == 0 else "normal")
                self._text(ax, x + w - 16, yy, f"{100 * p:.1f}%", 13, INK, ha="right")
                ax.add_patch(Rectangle((cx, yy + 18), x + w - 16 - cx, 5, fc=WASH, ec="none"))
                ax.add_patch(Rectangle((cx, yy + 18), (x + w - 16 - cx) * p, 5,
                                       fc=INK if k == 0 else FAINT, ec="none"))
            self._text(ax, cx, y + h - 22, "Policy probabilities, not chances of success.", 11,
                       MUTED)

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


def smart_times(record: dict, fps: int) -> list[float]:
    """Frame times for the whole day at smart speed: flights at 1 simulated minute per second,
    everything else at 8, each decision held for 0.6 s."""
    segs, decs, end = record["segments"], record["decisions"], record["meta"]["end"]
    times, t, d = [], 0.0, 0
    while t < end:
        while d < len(decs) and decs[d]["t"] <= t + EPS:  # a decision: hold it
            t = max(t, decs[d]["t"])
            times += [decs[d]["t"]] * round(HOLD_S * fps)
            d += 1
        state = state_at(record, t)
        flying = state["active"] is not None and segs[state["active"]][0] == "fly"
        t_next = t + (FLY_RATE if flying else GROUND_RATE) / fps
        if d < len(decs):
            t_next = min(t_next, decs[d]["t"])
        times.append(t)
        t = min(t_next, end) if t_next > t + EPS else end
    return times + [end] * fps


def explained_times(record: dict, first: int, count: int, fps: int,
                    before_s: float = 2.0, action_s: float = 3.0, after_s: float = 2.0) -> list:
    """(time, phase) for `count` decisions from index `first`: each shown before, during and
    after its action."""
    out = []
    for d in record["decisions"][first:first + count]:
        out += [(d["t"], "Deciding")] * round(before_s * fps)
        n = round(action_s * fps)
        out += [(d["t"] + (d["end"] - d["t"]) * k / n, None) for k in range(n)]
        out += [(d["end"] - 1e-6, "Done")] * round(after_s * fps)
    return out
