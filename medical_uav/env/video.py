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
# the whole-day comparison: simulated min/s by activity (the slower side sets the pace)
DAY_RATES, DAY_GROUND = {"fly": 4.0, "load": 5.0, "unload": 5.0, "dispose": 5.0}, 12.0


def pt(px: float) -> float:
    """Font size in points for a size in pixels."""
    return px * 72 / DPI


def rounded(x, y, w, h, radius, **style) -> FancyBboxPatch:
    """A rectangle with rounded corners."""
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", **style)


class Subtitles:
    """A band under each frame with a heading and up to two lines, in place of narration."""

    H = 200

    def __init__(self, bold: str, regular: str, width: int = 1920, sizes=(46, 36)):
        from PIL import ImageFont
        self.bold, self.reg = (ImageFont.truetype(bold, sizes[0]),
                               ImageFont.truetype(regular, sizes[1]))
        self.width, self.cache = width, {}

    def _wrap(self, text):
        lines, cur = [], ""
        for word in text.split(" "):
            if cur and self.reg.getlength(cur + " " + word) > self.width - 96:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        return lines + [cur] if cur else lines

    def band(self, title: str, body: str = "", paused: bool = False) -> np.ndarray:
        key = (title, body, paused)
        if key not in self.cache:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (self.width, self.H), (20, 28, 46))
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, 14, self.H), fill=(232, 89, 12))  # accent bar
            d.text((48, 14), title, font=self.bold, fill=(255, 196, 92) if paused else
                   (255, 255, 255))
            for j, line in enumerate(self._wrap(body)[:2]):
                d.text((48, 84 + 50 * j), line, font=self.reg, fill=(226, 232, 242))
            self.cache[key] = np.asarray(img)
        return self.cache[key]


def zoom(frame: np.ndarray, box, a: float, badge=None) -> np.ndarray:
    """The frame zoomed a (0..1) of the way from the whole frame into box (x, y, w, h; widened
    to the frame's shape), smoothly; with badge = (text, font) a pill at the bottom centre."""
    from PIL import Image, ImageDraw
    H_, W_ = frame.shape[:2]
    x, y, w, h = box
    if w / h < W_ / H_:
        x, w = x - (h * W_ / H_ - w) / 2, h * W_ / H_
    else:
        y, h = y - (w * H_ / W_ - h) / 2, w * H_ / W_
    x, y = min(max(x, 0), W_ - w), min(max(y, 0), H_ - h)
    e = a * a * (3 - 2 * a)
    crop = (e * x, e * y, e * x + (1 - e) * W_ + e * w, e * y + (1 - e) * H_ + e * h)
    img = Image.fromarray(frame).resize((W_, H_), Image.LANCZOS, box=crop)
    if badge:
        d = ImageDraw.Draw(img)
        text, font = badge
        tw = d.textlength(text, font=font)
        x0, y0 = (W_ - tw - 48) / 2, H_ - 88  # bottom centre
        d.rounded_rectangle((x0, y0, x0 + tw + 48, y0 + 64), 32, fill=(232, 89, 12))
        d.text((x0 + 24, y0 + 6), text, font=font, fill="white")
    return np.asarray(img)


PHASE_LABEL = {"Deciding": "SCORING ACTIONS · DECISION", "Picked": "ACTION CHOSEN · DECISION"}


def pct(p: float, digits: int = 1) -> str:
    """A probability in percent; tiny ones as <0.1%."""
    return "<0.1%" if p < 0.0005 else f"{100 * p:.{digits}f}%"


def more_text(d: dict) -> str:
    """The allowed actions beyond the listed top choices, with their summed probability."""
    n = d.get("allowed", len(d["top"])) - len(d["top"])
    if n <= 0:
        return ""
    rest = max(0.0, 1 - sum(p for _, p in d["top"]))
    share = "below 0.1%" if rest < 0.001 else f"{100 * rest:.1f}%"
    return f"… and {n} more allowed action{'s' if n > 1 else ''}, {share} in total"


def spotlight(fig, size, boxes, alpha, keep=()):
    """Dim a figure except `boxes` (pixel x, y, w, h; y downwards) and outline them; `keep` are
    further boxes left undimmed without an outline. alpha in [0, 1] fades the effect in."""
    W_, H_ = size
    over = fig.add_axes((0, 0, 1, 1), zorder=10)
    over.set(xlim=(0, W_), ylim=(H_, 0))
    over.axis("off")
    verts = [(0, 0), (W_, 0), (W_, H_), (0, H_), (0, 0)]
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    for x, y, w, h in list(boxes) + list(keep):  # holes, opposite winding
        verts += [(x, y), (x, y + h), (x + w, y + h), (x + w, y), (x, y)]
        codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    over.add_patch(PathPatch(Path(verts, codes), fc=PAPER, alpha=0.78 * alpha, ec="none"))
    for x, y, w, h in boxes:
        over.add_patch(rounded(x, y, w, h, 10, fill=False, ec="#e8590c", lw=3, alpha=alpha))


def clock(t: float, start_hour: int = 8) -> str:
    m = int(start_hour * 60 + t + 1e-9)
    return f"{m // 60:02d}:{m % 60:02d}"


class ReplayFrames:
    """Draws any minute of a recorded day."""

    def __init__(self, record: dict, subtitle: str = "", scale: float = 1.0):
        self.r, self.subtitle = record, subtitle
        self.orders = {o["id"]: o for o in record["orders"]}
        self.focus = {"nodes": (), "orders": ()}  # places and orders to highlight
        self.banner = ""  # a label over the replay, e.g. the scenario shown
        self.fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI * scale, facecolor=PAPER)
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
        return f"{-left:.0f} min overdue", LATE

    def _panel(self, ax, x, y, w, h):
        ax.add_patch(rounded(x, y, w, h, 10, facecolor=PANEL, edgecolor=LINE, lw=1))

    def _text(self, ax, x, y, s, size=13, color=INK, weight="normal", **kw):
        return ax.text(x, y, s, fontsize=pt(size), color=color, weight=weight, va="top", **kw)

    # ---------- frame ----------
    def draw(self, t: float, phase: str | None = None, speed: str = "",
             spot=None) -> np.ndarray:
        """The frame at time t; `phase` overrides the decision card's phase line; spot =
        (boxes, alpha) dims all but the boxes."""
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
        if spot:
            spotlight(self.fig, (W, H), spot[0], spot[1])
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()

    def _header(self, ax, s, t):
        meta = self._text(ax, 24, 16, "MEDICAL UAV · ONE SIMULATED DAY", 11, MUTED, "bold")
        meta.set_fontstretch("expanded")
        self._text(ax, 24, 32, self.r["meta"]["policy"], 22, INK, "bold")
        if self.banner:  # the scenario being shown, in place of the subtitle
            ax.add_patch(rounded(24, 58, 8.2 * len(self.banner) + 28, 26, 13, fc="#e8590c",
                                 ec="none"))
            self._text(ax, 38, 62, self.banner, 14, "white", "bold")
        else:
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
        for k in self.focus["nodes"]:
            n = nodes[k]
            frame.add_patch(Circle((n["x"], n["y"]), 0.3, fill=False, ec="#e9bf55", lw=4, zorder=5))
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
        label = "Scoring actions" if deciding else f"In progress · done at {self._clock(d['end'])}"
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
            shown = [round(100 * prob, 1) for _, prob in d["top"] if prob >= 0.0005]
            digits = 2 if len(set(shown)) < len(shown) else 1  # a near tie needs more digits
            self._text(ax, cx, y + 94, "THE NETWORK'S HIGHEST-PROBABILITY ACTIONS", 11, MUTED, "bold")
            for k, (label, prob) in enumerate(d["top"]):
                yy = y + 112 + 30 * k
                if k == 0 and name in ("Picked", "Moving", "Chosen"):
                    ax.add_patch(Rectangle((cx - 6, yy - 4), x + w - 10 - cx, 28, fc=DECIDING[0],
                                           ec="none"))
                self._text(ax, cx, yy, label, 13, INK, "bold" if k == 0 else "normal")
                self._text(ax, x + w - 16, yy, pct(prob, digits), 13, INK, ha="right")
                ax.add_patch(Rectangle((cx, yy + 18), x + w - 16 - cx, 5, fc=WASH, ec="none"))
                ax.add_patch(Rectangle((cx, yy + 18), (x + w - 16 - cx) * prob, 5,
                                       fc=INK if k == 0 else FAINT, ec="none"))
            self._text(ax, cx, y + h - 32, more_text(d), 12, MUTED)
            self._text(ax, cx, y + h - 15, "Action probabilities, not chances of success.", 11,
                       MUTED)
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
            hot = o is not None and o["id"] in self.focus["orders"]
            ax.add_patch(rounded(x + 16, yy, w - 32, 32, 6, fc=DECIDING[0] if hot else PANEL,
                                 ec="#e9bf55" if hot else LINE, lw=2 if hot else 1,
                                 ls="-" if o else "--"))
            if o:
                ax.add_patch(Rectangle((x + 28, yy + 11), 10, 10, fc=TYPE_COLORS[o["type"]]))
                nodes = r["map"]["nodes"]
                self._text(ax, x + 46, yy + 8,
                           f"{TYPE_LABELS[o['type']]} → {nodes[o['to']]['name']}", 13)
                due, color = self._due(o, t)
                self._text(ax, x + w - 28, yy + 9, due, 12, color, ha="right")
            else:
                self._text(ax, x + 46, yy + 8, "Free space", 13, FAINT)
        # waiting orders
        x2, w2 = 910, 346
        self._panel(ax, x2, y, w2, h)
        waiting = sorted((self.orders[i] for i in s["waiting"]),
                         key=lambda o: (o["deadline"], o["id"]))
        self._text(ax, x2 + 16, y + 14, "ORDERS AWAITING PICKUP", 11, MUTED, "bold")
        self._text(ax, x2 + w2 - 16, y + 14, f"{len(waiting)} waiting · most urgent first", 12,
                   MUTED, ha="right")
        nodes = r["map"]["nodes"]
        for k, o in enumerate(waiting[:6]):
            yy = y + 40 + 36 * k
            hot = o["id"] in self.focus["orders"]
            ax.add_patch(rounded(x2 + 16, yy, w2 - 32, 30, 6, fc=DECIDING[0] if hot else PAPER,
                                 ec="#e9bf55" if hot else "none", lw=2))
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
    orders; below, three running charts (on time so far, backlog, outcome bars)."""

    W, H = 1920, 1080
    colors = ("#e8590c", "#7048e8")  # the report's PPO and Slack-aware colours
    RULE = "Unload here first · take boxes going where cargo goes · charge if needed · " \
           "else earliest deadline first, too-late orders last"

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
    def draw(self, t: float, paused=False, phases=(None, None), speed=None) -> np.ndarray:
        """The frame at time t; phases: per side None or a step of a new decision, "Deciding"
        (the choices shown) then "Picked" (the top one taken)."""
        ax = self._canvas()
        meta = self.r["meta"]
        self._text(ax, 32, 22, "Same orders, two dispatchers", 34, INK, "bold")
        self._text(ax, 32, 70, f"{meta['load'].capitalize()} demand · "
                   "same map and same order arrivals for both", 17, MUTED)
        self._text(ax, self.W - 32, 14, clock(t, meta["day_start_hour"]), 52, INK, "bold",
                   ha="right")
        peak = any(a <= t < b for a, b in meta["peaks"])
        if speed:
            self._text(ax, self.W - 32, 76, speed, 15, MUTED, ha="right")
        if peak:
            ax.add_patch(rounded(1450, 30, 200, 40, 20, fc="#fff0c2", ec="#e9bf55"))
            self._text(ax, 1550, 39, "DEMAND PEAK", 17, "#6f4600", "bold", ha="center")
        if paused:
            ax.add_patch(rounded(1270, 30, 160, 40, 20, fc="#e8590c", ec="none"))
            self._text(ax, 1350, 39, "❚❚ PAUSED", 17, "white", "bold", ha="center")
        self._legend(ax, 720, 74)
        for k, view in enumerate(self.views):
            self._side(ax, view, state_at(view.r, min(t, view.r["meta"]["end"])), t, k,
                       phases[k])
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
        frame = self._box(x, 156, 380, 380)
        size = view.r["map"]["size"]
        frame.set(xlim=(-0.15, size + 0.15), ylim=(size + 0.15, -0.15))
        frame.axis("off")
        view._draw_map(frame, s, t)
        self._decision(ax, view, s, t, x + 392, color, phase)
        self._onboard(ax, view, s, t, x + 392)
        self._pool(ax, view, s, t, x)

    def _decision(self, ax, view, s, t, x, color, phase=None):
        y, w, h = 150, 520, 306
        self._panel(ax, x, y, w, h, **({"edgecolor": "#e9bf55", "lw": 3} if phase else {}))
        r = view.r
        i = s["decision"]
        d = r["decisions"][i] if i >= 0 else None
        if t >= r["meta"]["end"] - 1e-6:
            self._text(ax, x + 18, y + 14, "DAY COMPLETE", 14, MUTED, "bold")
            self._text(ax, x + 18, y + 36, "Every order delivered or expired", 23, INK, "bold")
            return
        if d is None:
            self._text(ax, x + 18, y + 14, "START OF THE DAY", 14, MUTED, "bold")
            return
        head = f"{PHASE_LABEL.get(phase, 'DECISION')} {i + 1} OF {len(r['decisions'])} · " \
               f"{view._clock(d['t'])}"
        self._text(ax, x + 18, y + 14, head, 14, "#6f4600" if phase else MUTED, "bold")
        lines = ["Scoring the allowed actions …"] if phase == "Deciding" else wrap(d["text"], 38)[:2]
        for j, line in enumerate(lines):
            self._text(ax, x + 18, y + 36 + 29 * j, line, 23,
                       FAINT if phase == "Deciding" else INK, "bold")
        fy = y + 78 + 29 * (len(lines) - 1)
        for j, fact in enumerate(d["facts"][:3]):
            ax.add_patch(Circle((x + 23, fy + 10 + 22 * j), 2.5, color=FAINT))
            self._text(ax, x + 32, fy + 22 * j, fact, 15)
        yy = fy + 74
        if d["top"]:
            self._text(ax, x + 18, yy, "THE NETWORK'S HIGHEST-PROBABILITY ACTIONS", 13, MUTED, "bold")
            for j, (label, prob) in enumerate(d["top"]):
                ry = yy + 20 + 24 * j
                if j == 0 and phase == "Picked":
                    ax.add_patch(Rectangle((x + 10, ry - 3), w - 20, 25, fc=DECIDING[0],
                                           ec="none"))
                self._text(ax, x + 18, ry, label, 15, INK, "bold" if j == 0 else "normal")
                self._text(ax, x + w - 18, ry, pct(prob), 15, INK, ha="right")
                ax.add_patch(Rectangle((x + 18, ry + 19), w - 36, 3, fc=WASH, ec="none"))
                ax.add_patch(Rectangle((x + 18, ry + 19), (w - 36) * prob, 3,
                                       fc=color if j == 0 else FAINT, ec="none"))
            self._text(ax, x + 18, yy + 22 + 24 * len(d["top"]), more_text(d), 13, MUTED)
        else:
            self._text(ax, x + 18, yy, "SLACK-AWARE RULE", 13, MUTED, "bold")
            for j, line in enumerate(wrap(self.RULE, 52)):
                self._text(ax, x + 18, yy + 20 + 22 * j, line, 15, INK)

    def _onboard(self, ax, view, s, t, x):
        y, w, h = 466, 520, 124
        self._panel(ax, x, y, w, h)
        r = view.r
        b = s["battery"]
        self._text(ax, x + 18, y + 12, "BATTERY", 13, MUTED, "bold")
        note = f"{b:.0f}%"
        if s["active"] is not None and r["segments"][s["active"]][0] == "charge":
            note = f"charging to {r['segments'][s['active']][6]:.0f}% · " + note
        self._text(ax, x + w - 18, y + 8, note, 16, INK, "bold", ha="right")
        bx, bw = x + 104, w - 104 - (240 if "charging" in note else 74)
        ax.add_patch(rounded(bx, y + 15, bw, 10, 5, fc=WASH, ec="none"))
        ax.add_patch(rounded(bx, y + 15, bw * b / 100, 10, 5,
                             fc=EXPIRED if b < r["meta"]["reserve"] + 10 else ONTIME, ec="none"))
        cargo = sorted((view.orders[o] for o in s["cargo"]), key=lambda o: o["pickup"])
        self._text(ax, x + 18, y + 40, f"ON BOARD · {len(cargo)} OF {r['meta']['max_boxes']} BOXES · "
                   f"{sum(o['kg'] for o in cargo):.2f}/{r['meta']['max_kg']:.2f} KG", 13, MUTED,
                   "bold")
        for j in range(r["meta"]["max_boxes"]):
            yy = y + 60 + 21 * j
            if j < len(cargo):
                self._order(ax, view, cargo[j], t, x + 18, yy, w - 36)
            else:
                self._text(ax, x + 38, yy, "free space", 14, FAINT)

    def _pool(self, ax, view, s, t, x):
        """Every waiting order, most urgent first, in three columns."""
        y, w, h, rows = 600, 912, 162, 6
        self._panel(ax, x, y, w, h)
        waiting = sorted((view.orders[o] for o in s["waiting"]),
                         key=lambda o: (o["deadline"], o["id"]))
        self._text(ax, x + 18, y + 10, f"ORDERS AWAITING PICKUP · {len(waiting)} · EARLIEST DEADLINE FIRST", 13,
                   MUTED, "bold")
        if not waiting:
            self._text(ax, x + 18, y + 34, "None", 14, FAINT)
        cw = (w - 36 - 32) / 3
        shown = waiting if len(waiting) <= 3 * rows else waiting[:3 * rows - 1]
        for j, o in enumerate(shown):
            cx, ry = x + 18 + (j // rows) * (cw + 16), y + 32 + 21 * (j % rows)
            ax.add_patch(rounded(cx - 6, ry - 2, cw + 12, 19, 4, fc=PAPER, ec="none"))
            self._order(ax, view, o, t, cx, ry, cw, 13)
        if len(shown) < len(waiting):
            self._text(ax, x + 18 + 2 * (cw + 16), y + 32 + 21 * (rows - 1),
                       f"+{len(waiting) - len(shown)} more", 13, MUTED, "bold")

    def _order(self, ax, view, o, t, x, y, w, size=14):
        nodes = view.r["map"]["nodes"]
        ax.add_patch(Rectangle((x, y + 3), size - 3, size - 3, fc=TYPE_COLORS[o["type"]]))
        self._text(ax, x + size + 4, y, f"#{o['id']} {TYPE_LABELS[o['type']]} "
                   f"{nodes[o['from']]['name']}→{nodes[o['to']]['name']}", size)
        left = o["deadline"] - t
        due, shade = ((f"due in {left:.0f} min", LATE if left < 10 else MUTED) if left >= 0 else
                      (f"{-left:.0f} min overdue", LATE))
        self._text(ax, x + w, y, due, size, shade, ha="right")

    def _legend(self, ax, x, y):
        """Colour key of the order squares."""
        self._text(ax, x, y + 2, "ORDER TYPES", 13, MUTED, "bold")
        x += 118
        for label, shade in zip(TYPE_LABELS, TYPE_COLORS):
            ax.add_patch(Rectangle((x, y + 3), 14, 14, fc=shade))
            self._text(ax, x + 20, y, label, 16, INK)
            x += 34 + 10 * len(label)

    def _charts(self, t):
        y, h = 776, 292
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
                c.plot(xs, ys, color=self.colors[k], lw=2.6, drawstyle="steps-post",
                       label=self.titles[k])
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
            c.legend(loc="upper left", frameon=False, fontsize=pt(14))
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

    # what each step of the screen guide lights up: pixel boxes, then its subtitle
    GUIDE = [([(26, 150, 392, 392), (970, 150, 392, 392)], "The map",
              ["M: medical center; H1–H5: hospitals; C1, C2: chargers; hatched: no-fly zones.",
               "Coloured squares: orders awaiting pickup there. (Paused at 10:30; the day then "
               "starts at 08:00.)"]),
             ([(418, 144, 532, 452), (1362, 144, 532, 452)], "What each dispatcher does now",
              ["The current action and the situation it was chosen in; PPO also shows its",
               "network's most likely actions. Below: battery and the boxes on board."]),
             ([(26, 594, 924, 174), (970, 594, 924, 174)], "Orders awaiting pickup",
              ["Orders that have arrived but are not picked up yet, earliest deadline first,",
               "with the minutes until each deadline."]),
             ([(352, 94, 592, 50), (1296, 94, 592, 50)], "The score so far",
              ["Orders delivered on time, delivered late, expired, and still waiting.", ""]),
             ([(26, 770, 1868, 304)], "Running totals for both",
              ["On-time deliveries so far, orders awaiting pickup (the backlog), and the counts",
               "of on-time, late and expired orders, for both dispatchers."])]

    def guide(self, t: float, step: int = 0) -> np.ndarray:
        """The frame at t, dimmed except the boxes of one guide step (GUIDE: boxes, title, text
        for the subtitles)."""
        self.draw(t)
        spotlight(self.fig, (self.W, self.H), self.GUIDE[step][0], 1.0)
        return self._rgb()

    def results(self, title, rows, note, takeaway="") -> np.ndarray:
        """A bar chart card; rows = (label, colour, medium %, high %, ms per decision)."""
        ax = self._canvas()
        self._panel(ax, 64, 56, self.W - 128, self.H - 112)
        self._text(ax, 110, 90, "RESULTS", 16, MUTED, "bold")
        self._text(ax, 110, 120, title, 38, INK, "bold")
        for k, part in enumerate(wrap(note, 130)):
            self._text(ax, 110, 172 + 28 * k, part, 20, MUTED)
        for j, load in enumerate(("Medium demand", "High demand")):
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
            yy = 270 + step * (i + 0.5) - 16
            self._text(ax, 110, yy - 10, row[0], 22, INK, "bold" if i == 0 else "normal")
            if len(row) > 5:
                self._text(ax, 110, yy + 17, row[5], 16, MUTED)
            hot = i < 2  # PPO and the rolling horizon: the comparison that matters
            ax.add_patch(rounded(1560, yy - 8, 180, 46, 23, fc=row[1] if hot else WASH,
                                 ec="none"))
            self._text(ax, 1650, yy, row[4], 23, "white" if hot else INK, "bold" if hot else
                       "normal", ha="center")
        self._text(ax, 1650, 236, "TIME PER DECISION", 17, INK, "bold", ha="center")
        for k, part in enumerate(wrap(takeaway, 100)):
            self._text(ax, 110, 925 + 36 * k, part, 26, INK, "bold")
        return self._rgb()

    def outline(self, items, current=None, note="") -> np.ndarray:
        """The outline slide, or a section slide with the current item lit."""
        ax = self._canvas()
        self._panel(ax, 96, 96, self.W - 192, self.H - 192)
        self._text(ax, 150, 150, "ME5418 GROUP 17 · MEDICAL DELIVERY DRONE", 18, MUTED, "bold")
        self._text(ax, 150, 192, "Outline" if current is None else items[current], 46, INK, "bold")
        if note:
            self._text(ax, 150, 266, note, 26, MUTED)
        for j, item in enumerate(items):
            y, lit = 380 + 84 * j, current is None or j == current
            ax.add_patch(Circle((176, y + 24), 24, fc=self.colors[0] if j == current else
                                (INK if lit else WASH)))
            ax.text(176, y + 24, str(j + 1), ha="center", va="center", fontsize=pt(22),
                    color="white" if lit else MUTED, weight="bold")
            self._text(ax, 224, y + 6, item, 30, INK if lit else FAINT,
                       "bold" if j == current else "normal")
        return self._rgb()

    # the task, one element per step: the map's four elements, then orders, the drone, the decision
    TASK = [("Medical center M", "Plasma and medicine are collected here; charger C1 too."),
            ("Five hospitals H1–H5", "Receive plasma and medicine; send samples and supplies."),
            ("A second charger C2", "Flying to a charger and charging both take time."),
            ("Three no-fly zones", "The drone must fly around the hatched areas."),
            ("Orders: one box each, with a time limit", ""),
            ("One drone", ""),
            ("The decision, again and again", "")]

    def task(self, step: int, t: float) -> np.ndarray:
        """The task built up element by element; from step 4 the map shows the day's state at t."""
        ax = self._canvas()
        self._text(ax, 72, 40, "1  The task", 44, INK, "bold")
        v = self.views[0]
        size, nodes = v.r["map"]["size"], v.r["map"]["nodes"]
        f = self._box(56, 150, 760, 760)
        f.set(xlim=(-0.15, size + 0.15), ylim=(size + 0.15, -0.15))
        f.axis("off")
        if step >= 4:
            s = state_at(v.r, t)
            v._draw_map(f, s, t)
            if step == 5:
                f.add_patch(Circle(s["xy"], 0.34, fill=False, ec="#e8590c", lw=5, zorder=6))
        else:
            roles = {"center"} | ({"hospital"} if step >= 1 else set()) | (
                {"charger"} if step >= 2 else set())
            part = {**v.r, "segments": [], "orders": [],
                    "map": {**v.r["map"], "nodes": [n for n in nodes if n["role"] in roles],
                            "zones": v.r["map"]["zones"] if step >= 3 else []}}
            ReplayFrames(part)._draw_map(f, {"active": None, "waiting": [], "cargo": [],
                                             "xy": (-9, -9)}, -10)
        for text in f.texts:  # the map is drawn larger than in the replay
            text.set_fontsize(text.get_fontsize() * 1.5)
        x, w = 870, 990
        for j, (head, line) in enumerate(self.TASK[:min(step, 3) + 1]):  # the map's elements
            y, now = 140 + 88 * j, j == step
            self._text(ax, x, y, head, 32, INK if now else MUTED, "bold")
            self._text(ax, x, y + 44, line, 26, INK if now else MUTED)

        def card(y, h, now, accent=False):
            self._panel(ax, x - 18, y, w, h, facecolor="#fff4e6" if accent else PANEL,
                        edgecolor="#e8590c" if now or accent else LINE, lw=3 if now else 1)
        if step >= 4:  # orders and their deadlines, colour-coded like the squares on the map
            card(500, 196, step == 4)
            self._text(ax, x + 6, 512, self.TASK[4][0], 32, INK, "bold")
            limits = {tp["name"]: tp["limit"] for tp in v.r["types"]}
            for k, (label, shade) in enumerate(zip(TYPE_LABELS, TYPE_COLORS)):
                cx = x + 6 + 238 * k
                ax.add_patch(Rectangle((cx, 570), 26, 26, fc=shade))
                self._text(ax, cx + 36, 564, label, 26, INK, "bold")
                self._text(ax, cx + 36, 600, f"{limits[label.lower()]:.0f} min", 26, INK)
            self._text(ax, x + 6, 650, "Late still counts, for less; expires at twice its limit.",
                       25, INK)
        if step >= 5:  # the drone's limits
            card(710, 160, step == 5)
            self._text(ax, x + 6, 722, self.TASK[5][0], 32, INK, "bold")
            meta = v.r["meta"]
            for k in range(meta["max_boxes"]):
                ax.add_patch(rounded(x + 6 + 38 * k, 778, 30, 30, 4, fc=TYPE_COLORS[1], ec="none"))
            self._text(ax, x + 134, 774, f"at most {meta['max_boxes']} boxes", 26, INK)
            self._text(ax, x + 430, 774, f"at most {meta['max_kg']:g} kg", 26, INK)
            ax.add_patch(rounded(x + 6, 830, 300, 18, 9, fc=WASH, ec="none"))
            ax.add_patch(rounded(x + 6, 830, 300 * 0.62, 18, 9, fc=ONTIME, ec="none"))
            rx = x + 6 + 300 * meta["reserve"] / 100
            ax.plot([rx, rx], [824, 854], color=INK, lw=2)
            self._text(ax, x + 330, 822, f"always able to reach a charger with {meta['reserve']:g}%"
                       " left", 24, INK)
        if step >= 6:  # what the dispatcher decides at every step
            card(884, 186, True, accent=True)
            self._text(ax, x + 6, 894, self.TASK[6][0], 32, "#b34700", "bold")
            self._text(ax, x + 6, 936, "Which order to pick up or deliver next?", 26, INK)
            self._text(ax, x + 6, 972, "When to charge, and to 60, 80 or 100%?", 26, INK)
            self._text(ax, x + 6, 1018, "Goal: as many orders on time as possible.", 28, INK,
                       "bold")
        return self._rgb()

    def method(self, title, boxes, lines) -> np.ndarray:
        """A left-to-right flow of boxes (heading, line) with arrows, and lines under it."""
        ax = self._canvas()
        self._text(ax, 96, 50, title, 44, INK, "bold")
        n, gap = len(boxes), 76
        w = (1728 - gap * (n - 1)) / n
        for k, (head, line) in enumerate(boxes):
            x = 96 + k * (w + gap)
            last = k == n - 1
            self._panel(ax, x, 180, w, 400, facecolor="#fff4e6" if last else PANEL,
                        edgecolor="#e8590c" if last else LINE, lw=3 if last else 1.5)
            heads = wrap(head, int(w / 19))
            for j, part in enumerate(heads):
                self._text(ax, x + 26, 206 + 44 * j, part, 31, "#b34700" if last else INK, "bold")
            for j, part in enumerate(wrap(line, int(w / 15))):
                self._text(ax, x + 26, 230 + 44 * len(heads) + 40 * j, part, 26, INK)
            if not last:
                ax.annotate("", (x + w + gap - 8, 380), (x + w + 8, 380),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=3, mutation_scale=34))
        y = 650
        for line in lines:
            for part in wrap(line, 88):
                self._text(ax, 96, y, part, 30, INK)
                y += 46
            y += 40
        return self._rgb()

    def bullets(self, title, lines, kicker, highlight="", size=29) -> np.ndarray:
        """A heading with numbered points, and optionally a highlighted box under them."""
        ax = self._canvas()
        self._panel(ax, 96, 96, self.W - 192, self.H - 192)
        self._text(ax, 150, 150, kicker, 18, MUTED, "bold")
        self._text(ax, 150, 192, title, 46, INK, "bold")
        y = 310
        for j, line in enumerate(lines):
            ax.add_patch(Circle((176, y + 22), 22, fc=self.colors[0]))
            ax.text(176, y + 22, str(j + 1), ha="center", va="center", fontsize=pt(20),
                    color="white", weight="bold")
            width, step = int(92 * 29 / size), round(42 * size / 29)
            for k, part in enumerate(wrap(line, width)):
                self._text(ax, 224, y + 2 + step * k, part, size, INK)
            y += step * len(wrap(line, width)) + round(36 * size / 29)
        if highlight:
            parts = wrap(highlight, 90)
            ax.add_patch(rounded(150, y + 4, self.W - 300, 40 * len(parts) + 40, 14,
                                 fc="#fff4e6", ec="#e8590c", lw=3))
            for k, part in enumerate(parts):
                self._text(ax, 180, y + 24 + 40 * k, part, 29, "#b34700", "bold")
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


def day_times(records, fps, end) -> list[float]:
    """Frame times for the whole day on one clock: each frame advances by the slower of the two
    sides' rates (DAY_RATES by activity, else DAY_GROUND simulated min/s)."""
    def rate(r, t):
        a = state_at(r, min(t, r["meta"]["end"]))["active"]
        kind = r["segments"][a][0] if a is not None else "wait"
        return DAY_RATES.get(kind, DAY_GROUND)
    times, t = [0.0], 0.0
    while t < end - EPS:
        t = min(end, t + min(rate(r, t) for r in records) / fps)
        times.append(t)
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
