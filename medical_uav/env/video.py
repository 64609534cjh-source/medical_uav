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

from medical_uav.env.config import EPS, ORDER_TYPES
from medical_uav.env.replay import state_at

W, H, DPI = 1280, 720, 80
PAPER, PANEL, INK, MUTED, FAINT = "#f3f6f9", "#ffffff", "#13213c", "#44526a", "#7f8da3"
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
DAY_RATES, DAY_GROUND = {"fly": 5.6, "load": 6.4, "unload": 6.4, "dispose": 6.4}, 14.4


def pt(px: float) -> float:
    """Font size in points for a size in pixels."""
    return px * 72 / DPI


def rounded(x, y, w, h, radius, **style) -> FancyBboxPatch:
    """A rectangle with rounded corners."""
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", **style)


class Band:
    """The title band above each frame: a dark navy strip (the colour of the video's text)
    with a warm yellow heading. A new title slides in from the left while the strip
    brightens."""

    H = 100
    BG, END, LIT = (28, 40, 66), (70, 88, 118), (58, 80, 124)
    SHADE = 1 - 0.25 * np.linspace(1, 0, 28)[:, None, None] ** 2

    @classmethod
    def shade(cls, picture):
        """picture with the band's soft shadow on its top rows."""
        out = picture.copy()
        out[:28] = (out[:28] * cls.SHADE).astype(np.uint8)
        return out

    def __init__(self, bold: str, width: int = 1920, index: int = 0):
        from PIL import ImageFont
        self.head = ImageFont.truetype(bold, 48, index=index)
        self.width, self.cache = width, {}

    def draw(self, title, appear=1.0, flash=0.0) -> np.ndarray:
        e = appear * appear * (3 - 2 * appear)
        key = (title, round(e, 2), round(flash, 2))
        if key not in self.cache:
            from PIL import Image, ImageDraw
            lit = lambda c: np.array([a + (b - a) * flash for a, b in zip(c, self.LIT)])
            t = np.linspace(0, 1, self.width)[None, :, None]
            c0, c1 = lit(self.BG), lit(self.END)
            img = Image.fromarray(np.broadcast_to(c0 + (c1 - c0) * t, (self.H, self.width, 3))
                                  .astype(np.uint8).copy())
            d = ImageDraw.Draw(img)
            shift = round((1 - e) * 60)
            bg = tuple(round(c) for c in c0)
            mix = lambda col: tuple(round(c + (b - c) * (1 - e)) for c, b in zip(col, bg))
            d.text((52 - shift, 16), title, font=self.head, fill=mix((255, 196, 92)))
            if title:
                d.rectangle((52 - shift, 76, 172 - shift, 80), fill=mix((255, 196, 92)))
            self.cache[key] = np.asarray(img)
        return self.cache[key]


class Captions:
    """Narration drawn into the frame: a dark box with a heading and up to two lines. PLACES
    gives, per scene, the box's x, width, heading and line sizes and whether it hangs from the
    top or stands on the bottom of the frame. A new line slides in and its border flashes."""

    PLACES = {"slide": (96, 1728, 34, 28, "bottom"), "replay": (36, 1848, 34, 28, "bottom"),
              "duel": (24, 1256, 30, 24, "top"), "zoom_top": (60, 1800, 40, 32, "top"),
              "zoom_bottom": (60, 1800, 40, 32, "bottom"), "card": (60, 1800, 44, 36, "bottom")}
    MARGIN = 10

    def __init__(self, bold: str, regular: str, index: int = 0):
        from PIL import ImageFont
        self.font = lambda path, size: ImageFont.truetype(path, size, index=index)
        self.paths, self.cache = (bold, regular), {}

    @staticmethod
    def wrap(text, font, width):
        """Lines of at most width pixels: English breaks between words, Chinese anywhere."""
        import re
        lines, cur = [], ""
        for token in re.findall(r"[\u2e80-\u9fff\uff00-\uffef]|[^\s\u2e80-\u9fff\uff00-\uffef]+|\s+",
                                text.replace("\u00a0", " ")):
            if cur and not token.isspace() and font.getlength(cur + token) > width:
                lines.append(cur.rstrip())
                cur = token
            elif cur or not token.isspace():
                cur += token
        return lines + [cur.rstrip()] if cur.strip() else lines

    def _patch(self, title, body, paused, place):
        key = (title, body, paused, place)
        if key not in self.cache:
            from PIL import Image, ImageDraw
            x, w, hs, bs, _ = self.PLACES[place]
            bold, reg = self.font(self.paths[0], hs), self.font(self.paths[1], bs)
            lines = self.wrap(body, reg, w - 64)[:2]
            head, step = round(hs * 1.3), round(bs * 1.4)
            h = 14 + head + step * len(lines) + 12
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            # solid navy card: the band's colour without its gradient
            d.rounded_rectangle((0, 0, w - 1, h - 1), 18, fill=(28, 40, 66, 245))
            d.text((34, 10), title, font=bold, fill=(255, 196, 92))
            for j, line in enumerate(lines):
                d.text((34, 14 + head + step * j), line, font=reg, fill=(255, 255, 255))
            self.cache[key] = np.asarray(img).astype(np.float32) / 255
        return self.cache[key]

    def height(self, title, body, place):
        return self._patch(title, body, False, place).shape[0]

    def draw(self, frame, title, body="", paused=False, place="slide", appear=1.0, flash=0.0):
        """The frame with the caption; appear (0..1) slides it in from its edge, flash (0..1)
        rings it orange."""
        if not title:
            return frame
        patch = self._patch(title, body, paused, place)
        x, w, _, _, anchor = self.PLACES[place]
        h = patch.shape[0]
        e = appear * appear * (3 - 2 * appear)
        if anchor == "top":
            y = int(self.MARGIN - (1 - e) * 28)
        else:
            y = int(frame.shape[0] - self.MARGIN - h + (1 - e) * 28)
        top, bottom = max(y, 0), min(y + h, frame.shape[0])
        part = patch[top - y:bottom - y]
        out = frame.astype(np.float32)
        region = out[top:bottom, x:x + w]
        alpha = part[..., 3:] * e
        region[:] = region * (1 - alpha) + part[..., :3] * 255 * alpha
        if flash > 0:  # an orange ring around the box
            ring = np.zeros(part.shape[:2] + (1,), np.float32)
            ring[:, :5] = ring[:, -5:] = 1
            if top == y:
                ring[:5] = 1
            if bottom == y + h:
                ring[-5:] = 1
            ring *= part[..., 3:] * flash * e  # the ring appears with the box
            region[:] = region * (1 - ring) + np.array([255, 196, 92], np.float32) * ring
        return out.astype(np.uint8)


def _view(box, W_, H_):
    """box (x, y, w, h) widened to the frame's shape and kept inside it, as a crop rectangle."""
    if box is None:
        return 0, 0, W_, H_
    x, y, w, h = box
    if w / h < W_ / H_:
        x, w = x - (h * W_ / H_ - w) / 2, h * W_ / H_
    else:
        y, h = y - (w * H_ / W_ - h) / 2, w * H_ / W_
    x, y = min(max(x, 0), W_ - w), min(max(y, 0), H_ - h)
    return x, y, x + w, y + h


def morph(a_frame, b_frame, a_boxes, b_boxes, p: float) -> np.ndarray:
    """A p (0..1) smooth transition from a_frame to b_frame: the whole picture cross-fades
    while each region a_boxes[k] of the first moves and grows into b_boxes[k] of the second."""
    from PIL import Image
    e = p * p * (3 - 2 * p)
    out = Image.fromarray((a_frame * (1 - e) + b_frame * e).astype(np.uint8))
    for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(a_boxes, b_boxes):
        x, y, w, h = (round(u + (v - u) * e) for u, v in
                      zip((ax, ay, aw, ah), (bx, by, bw, bh)))
        part = lambda f, X, Y, W_, H_: np.asarray(Image.fromarray(f[Y:Y + H_, X:X + W_])
                                                  .resize((w, h), Image.LANCZOS), np.float32)
        piece = part(a_frame, ax, ay, aw, ah) * (1 - e) + part(b_frame, bx, by, bw, bh) * e
        out.paste(Image.fromarray(piece.astype(np.uint8)), (x, y))
    return np.asarray(out)


def zoom(frame: np.ndarray, box, a: float, badge=None, start=None) -> np.ndarray:
    """The view moved a (0..1) of the way, smoothly, from `start` (a box, or the whole frame)
    into box (x, y, w, h); with badge = (text, font) a pill at the bottom centre."""
    from PIL import Image, ImageDraw
    H_, W_ = frame.shape[:2]
    e = a * a * (3 - 2 * a)
    crop = tuple(p + (q - p) * e for p, q in zip(_view(start, W_, H_), _view(box, W_, H_)))
    img = Image.fromarray(frame).resize((W_, H_), Image.LANCZOS, box=crop)
    if badge:
        d = ImageDraw.Draw(img)
        text, font = badge
        tw = d.textlength(text, font=font)
        x0, y0 = (W_ - tw - 48) / 2, H_ - 88  # bottom centre
        d.rounded_rectangle((x0, y0, x0 + tw + 48, y0 + 64), 32, fill=(232, 89, 12))
        d.text((x0 + 24, y0 + 6), text, font=font, fill="white")
    return np.asarray(img)


SLIDE_BOTTOM = 912  # slide content ends here; the caption box stands below it
PHASE_LABEL = {"Deciding": "SCORING ACTIONS · DECISION", "Picked": "ACTION CHOSEN · DECISION"}


WEIGHTS = {t.name: t.weight_range for t in ORDER_TYPES}  # kg per box, drawn at random


def flown(r: dict, t: float, empty: bool = False) -> float:
    """Kilometres the drone has flown by time t; with empty, only those flown with no box on
    board."""
    km = 0.0
    for s in r["segments"]:
        if s[0] == "fly" and s[3] != s[4] and s[1] < t and not (
                empty and state_at(r, s[1] + EPS)["cargo"]):
            path = np.asarray(r["map"]["routes"][f"{s[3]}-{s[4]}"])
            km += np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * min(
                1.0, (t - s[1]) / max(s[2] - s[1], EPS))
    return km


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
        self.timeline = True  # the day's timeline at the bottom (off leaves room for captions)
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
             spot=None, decision: int | None = None) -> np.ndarray:
        """The frame at time t; `phase` overrides the decision card's phase line; spot =
        (boxes, alpha) dims all but the boxes; `decision` keeps the card on that decision (at
        the instant an action ends, the next decision has already begun)."""
        r, s = self.r, state_at(self.r, t)
        if decision is not None:
            s = {**s, "decision": decision}
        self.fig.clear()
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        self._header(ax, s, t)
        self._map(s, t)
        self._decision(ax, s, t, phase)
        self._state(ax, s, t)
        if self.timeline:
            self._timeline(ax, t, speed)
        elif speed:  # without the timeline the speed note goes under the clock
            self._text(ax, W - 24, 74, speed, 11, MUTED, ha="right")
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
        self._text(ax, x2 + w2 - 16, y + 14, f"{len(waiting)} · by deadline", 12,
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

    def __init__(self, left, right, titles=("PPO agent", "Slack-aware rule")):
        self.views = (ReplayFrames(left), ReplayFrames(right))
        self.titles, self.r = titles, left
        self.fig = Figure(figsize=(self.W / DPI, self.H / DPI), dpi=DPI, facecolor=PAPER)
        self.canvas = FigureCanvasAgg(self.fig)
        self.end = max(v.r["meta"]["end"] for v in self.views)
        minutes = np.arange(0, self.end + 0.5, 0.5)
        self.minutes = minutes
        self.series = []  # per side: on time so far, orders waiting, empty km flown so far
        for v in self.views:
            states = [state_at(v.r, min(m, v.r["meta"]["end"])) for m in minutes]
            self.series.append((np.array([s["on_time"] for s in states]),
                                np.array([len(s["waiting"]) for s in states]),
                                np.array([flown(v.r, m, True) for m in minutes])))

    # ---------- helpers ----------
    def _text(self, ax, x, y, s, size=16, color=INK, weight="normal", **kw):
        return ax.text(x, y, s, fontsize=pt(size), color=color, weight=weight, va="top", **kw)

    def _panel(self, ax, x, y, w, h, **style):
        ax.add_patch(rounded(x, y, w, h, 12, **{"facecolor": PANEL, "edgecolor": LINE, "lw": 1,
                                                **style}))

    def _page(self, ax):
        """A slide: the whole frame white, with a thin orange line at the top left."""
        ax.add_patch(Rectangle((0, 0), self.W, self.H, fc=PANEL, ec="none"))
        ax.add_patch(Rectangle((96, 60), 140, 8, fc=self.colors[0], ec="none"))

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
    def draw(self, t: float, paused=False, phases=(None, None)) -> np.ndarray:
        """The frame at time t; phases: per side None or a step of a new decision, "Deciding"
        (the choices shown) then "Picked" (the top one taken)."""
        ax = self._canvas()
        meta = self.r["meta"]
        self._text(ax, 32, 30, "Same orders, two dispatchers", 34, INK, "bold")
        self._text(ax, self.W - 32, 8, clock(t, meta["day_start_hour"]), 52, INK, "bold",
                   ha="right")
        peak = any(a <= t < b for a, b in meta["peaks"])
        if peak:
            ax.add_patch(rounded(1300, 18, 190, 40, 20, fc="#fff0c2", ec="#e9bf55"))
            self._text(ax, 1395, 27, "DEMAND PEAK", 17, "#6f4600", "bold", ha="center")
        if paused:
            ax.add_patch(rounded(1500, 18, 170, 40, 20, fc="#e8590c", ec="none"))
            self._text(ax, 1585, 27, "❚❚ PAUSED", 17, "white", "bold", ha="center")
        self._legend(ax, 1300, 98)
        for k, view in enumerate(self.views):
            self._side(ax, view, state_at(view.r, min(t, view.r["meta"]["end"])), t, k,
                       phases[k])
        self._charts(t)
        return self._rgb()

    def _side(self, ax, view, s, t, k, phase=None):
        """One dispatcher: name and score, map, and beside it what it is doing."""
        x, color = 32 + 944 * k, self.colors[k]
        self._text(ax, x, 132, self.titles[k], 28, color, "bold")
        chips = (("on time", s["on_time"], ONTIME), ("late", s["late"], LATE),
                 ("expired", s["expired"], EXPIRED), ("waiting", len(s["waiting"]), MUTED))
        cx = x + 328
        for label, n, shade in chips:
            ax.add_patch(rounded(cx, 132, 138, 38, 19, fc=PANEL, ec=LINE, lw=1))
            ax.add_patch(Circle((cx + 19, 151), 7, color=shade))
            self._text(ax, cx + 34, 137, str(n), 20, INK, "bold")
            self._text(ax, cx + 40 + 14 * len(str(n)), 141, label, 14, MUTED)
            cx += 146
        frame = self._box(*self.duel_map(k))
        size = view.r["map"]["size"]
        frame.set(xlim=(-0.15, size + 0.15), ylim=(size + 0.15, -0.15))
        frame.axis("off")
        view._draw_map(frame, s, t)
        for text in frame.texts:
            text.set_fontsize(text.get_fontsize() * 1.2)
        self._status(ax, view, s, t, x + 576, 188, 336, 560, {}, color, phase)

    def _status(self, ax, view, s, t, x, y, w, h, tags, color, phase=None):
        """What a dispatcher is doing, with few words: its action now (for PPO with the
        network's probability for it), battery and the box slots, and the most urgent waiting
        orders with the minutes left; the orders in tags are highlighted."""
        r, nodes = view.r, view.r["map"]["nodes"]

        def dot(cx, cy, o, size):
            ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size,
                                   fc=TYPE_COLORS[o["type"]],
                                   ec=LATE if t > o["deadline"] else "none", lw=2))

        d = r["decisions"][s["decision"]] if s["decision"] >= 0 else None
        done = t >= r["meta"]["end"] - 1e-6
        self._panel(ax, x, y, w, 122, **({"edgecolor": "#e9bf55", "lw": 3} if phase else {}))
        self._text(ax, x + 16, y + 10, "NOW", 13, MUTED, "bold")
        text = "Day complete" if done else d["text"] if d else "Start of the day"
        for j, line in enumerate(wrap(text, int(w / 13))[:2]):
            self._text(ax, x + 16, y + 30 + 28 * j, line, 21, INK, "bold")
        if d and d["top"] and not done:  # the network's probability for the action taken
            prob = d["top"][0][1]
            ax.add_patch(rounded(x + 16, y + 98, w - 150, 8, 4, fc=WASH, ec="none"))
            ax.add_patch(rounded(x + 16, y + 98, (w - 150) * prob, 8, 4, fc=color, ec="none"))
            self._text(ax, x + w - 16, y + 91, f"network {pct(prob)}", 13, MUTED, ha="right")

        y2 = y + 132
        self._panel(ax, x, y2, w, 120)
        b = s["battery"]
        self._text(ax, x + 16, y2 + 10, "BATTERY · BOXES", 13, MUTED, "bold")
        self._text(ax, x + w - 16, y2 + 6, f"{b:.0f}%", 16, INK, "bold", ha="right")
        ax.add_patch(rounded(x + 150, y2 + 14, w - 220, 10, 5, fc=WASH, ec="none"))
        ax.add_patch(rounded(x + 150, y2 + 14, (w - 220) * b / 100, 10, 5,
                             fc=EXPIRED if b < r["meta"]["reserve"] + 10 else ONTIME, ec="none"))
        cargo = sorted((view.orders[o] for o in s["cargo"]), key=lambda o: o["pickup"])
        n = r["meta"]["max_boxes"]
        sw = (w - 32 - 10 * (n - 1)) / n
        for j in range(n):
            bx = x + 16 + j * (sw + 10)
            if j < len(cargo):
                o = cargo[j]
                hot = o["id"] in tags
                ax.add_patch(rounded(bx, y2 + 44, sw, 64, 8, fc=TYPE_COLORS[o["type"]],
                                     ec="#f2b705" if hot else "none", lw=5))
                self._text(ax, bx + sw / 2, y2 + 48, TYPE_LABELS[o["type"]], 14, "white",
                           ha="center")
                self._text(ax, bx + sw / 2, y2 + 70, f"to {nodes[o['to']]['name']}", 20, "white",
                           "bold", ha="center")
            else:
                ax.add_patch(rounded(bx, y2 + 44, sw, 64, 8, fc=PANEL, ec=LINE, ls="--"))
                self._text(ax, bx + sw / 2, y2 + 66, "empty", 14, FAINT, ha="center")

        y3 = y2 + 130
        h3 = h - (y3 - y)
        self._panel(ax, x, y3, w, h3)
        waiting = sorted((view.orders[o] for o in s["waiting"]), key=lambda o: o["deadline"])
        self._text(ax, x + 16, y3 + 10, f"WAITING ORDERS · {len(waiting)} · MOST URGENT FIRST", 13,
                   MUTED, "bold")
        rows = int((h3 - 70) // 34)
        shown = waiting if len(waiting) <= rows else waiting[:rows - 1]
        for j, o in enumerate(shown):
            ry = y3 + 50 + 34 * j
            if o["id"] in tags:
                ax.add_patch(rounded(x + 8, ry - 13, w - 16, 31, 6, fc="#fff1c2", ec="#f2b705",
                                     lw=2))
            dot(x + 26, ry + 2, o, 18)
            self._text(ax, x + 44, ry - 9, f"{TYPE_LABELS[o['type']]}  "
                       f"{nodes[o['from']]['name']}→{nodes[o['to']]['name']}", 17,
                       INK, "bold" if o["id"] in tags else "normal")
            left = o["deadline"] - t
            self._text(ax, x + w - 16, ry - 9, f"{left:.0f} min" if left >= 0 else "late", 17,
                       LATE if left < 10 else MUTED, "bold" if left < 10 else "normal", ha="right")
        if len(shown) < len(waiting):
            self._text(ax, x + 44, y3 + 50 + 34 * len(shown) - 9, f"+{len(waiting) - len(shown)} more",
                       15, MUTED)

    def _legend(self, ax, x, y):
        """Colour key of the order squares."""
        self._text(ax, x, y + 2, "ORDER TYPES", 13, MUTED, "bold")
        x += 118
        for label, shade in zip(TYPE_LABELS, TYPE_COLORS):
            ax.add_patch(Rectangle((x, y + 3), 14, 14, fc=shade))
            self._text(ax, x + 20, y, label, 16, INK)
            x += 34 + 10 * len(label)

    def _charts(self, t):
        """Below the two sides: orders delivered on time so far and orders waiting."""
        ax = self.fig.axes[0]
        for col, (x, title) in enumerate(((32, "Orders delivered on time so far"),
                                          (972, "Orders waiting now"))):
            self._panel(ax, x, 806, 916, 262)
            self._text(ax, x + 18, 816, title, 16, INK, "bold")
            self._line(self._box(x + 50, 846, 842, 192), col, t, 0.0, self.end)

    def _line(self, c, col, t, a, b, minus_start=False):
        """Series col of both sides from a up to t on axes c spanning a..b; minus_start counts
        from a."""
        m = self.minutes
        keep = (m >= a) & (m < t)
        at = lambda vals, x: vals[max(0, np.searchsorted(m, x + EPS) - 1)]  # value at minute x
        nows = []
        for k, series in enumerate(self.series):
            vals, r = series[col], self.views[k].r
            exact = lambda x: flown(r, x, True) if col == 2 else at(vals, x)  # km: no grid error
            base = exact(a) if minus_start else 0.0
            now = exact(t) - base
            c.plot(np.r_[m[keep], t], np.r_[vals[keep] - base, now], color=self.colors[k],
                   lw=2.8, drawstyle="steps-post", label=self.titles[k])
            c.plot(t, now, "o", color=self.colors[k], ms=7)
            nows.append(now)
        for k, now in enumerate(nows):
            above = now > nows[1 - k] or (now == nows[1 - k] and k == 0)
            c.annotate(f"{now:.0f}" if col < 2 else f"{now:.1f}", (t, now),
                       xytext=(8, 4 if above else -18), textcoords="offset points",
                       fontsize=pt(16), color=self.colors[k], weight="bold")
        top = max((s_[col][(m >= a) & (m <= b)] - (at(s_[col], a) if minus_start else 0)).max()
                  for s_ in self.series)
        for p, q in self.r["meta"]["peaks"]:
            c.axvspan(p, q, color=WASH, lw=0)
        ticks = np.arange(np.ceil(a / 60) * 60, b + 1, 120 if b - a > 240 else 10)
        c.set(xlim=(a, b + 0.06 * (b - a)), ylim=(0, top * 1.15 + 1))
        c.set_xticks(ticks, [clock(v) for v in ticks])
        c.tick_params(labelsize=pt(13), colors=MUTED, length=0)
        for side in ("top", "right"):
            c.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            c.spines[side].set_color(LINE)
        c.grid(axis="y", color=WASH, lw=1)
        c.set_facecolor(PANEL)
        c.legend(loc="upper left", frameon=False, fontsize=pt(14))

    @staticmethod
    def duel_map(k):
        """Pixel box of side k's map in the side-by-side frame."""
        return 32 + 944 * k, 188, 560, 560

    @staticmethod
    def close_map(k):
        """Pixel box of side k's map in the close-up."""
        return 40 + 940 * k, 158, 580, 580

    def close(self, t: float, since: float, focus=((), ()), ids=(), badge="",
              label="Close-up", until=None, spot=None, note="", at=None) -> np.ndarray:
        """Both maps enlarged side by side, each with its drone's route since `since` (solid
        with boxes on board, dashed empty) and, beside it, the current action, the cargo and
        the waiting orders; focus = per side the places to ring; ids = (order id, label) pairs:
        those orders are highlighted in the lists (the labels serve the report's figure, see
        tags); badge: a label beside the clock. Below, the stretch since..until as two small
        charts: orders on time and empty km flown since `since`. spot: pixel boxes to light up,
        with a callout holding note at `at` (default: under the first box)."""
        ax = self._canvas()
        meta = self.r["meta"]
        self._text(ax, 32, 22, f"{label} · same arrivals; each run continues from 08:00", 34,
                   INK, "bold")
        self._text(ax, self.W - 32, 8, clock(t, meta["day_start_hour"]), 52, INK, "bold",
                   ha="right")
        if badge:
            ax.add_patch(rounded(1450, 18, 220, 40, 20, fc="#e8590c", ec="none"))
            self._text(ax, 1560, 27, badge, 17, "white", "bold", ha="center")
        tc = max(t, since)  # counts start at since; before it they stay at zero
        for k, view in enumerate(self.views):
            r, x = view.r, 40 + 940 * k
            s, s0 = state_at(r, min(t, r["meta"]["end"])), state_at(r, since)
            self._text(ax, x, 112, self.titles[k], 28, self.colors[k], "bold")
            sc = state_at(r, min(tc, r["meta"]["end"]))
            self._text(ax, x + 900, 118, f"since {clock(since)}: {sc['on_time'] - s0['on_time']} "
                       f"on time · {flown(r, tc) - flown(r, since):.1f} km", 20, INK, "bold",
                       ha="right")
            frame = self._box(*self.close_map(k))
            size = r["map"]["size"]
            frame.set(xlim=(-0.15, size + 0.15), ylim=(size + 0.15, -0.15))
            frame.axis("off")
            view.focus = {"nodes": focus[k], "orders": ()}
            view._draw_map(frame, s, t)
            view.focus = {"nodes": (), "orders": ()}
            for text in frame.texts:
                text.set_fontsize(text.get_fontsize() * 1.4)
            self.trail(frame, r, since, t, self.colors[k])
            self._status(ax, view, s, t, x + 594, 158, 306, 580, dict(ids), self.colors[k])
        if until is not None:
            for col, (x, title) in ((0, (40, "Orders on time in this stretch")),
                                    (2, (980, "Empty kilometres in this stretch"))):
                self._panel(ax, x, 770, 900, 290)
                self._text(ax, x + 18, 780, title, 16, INK, "bold")
                self._line(self._box(x + 50, 812, 826, 218), col, tc, since, until, True)
        if spot:
            spotlight(self.fig, (self.W, self.H), spot, 1.0)
            x, y, w, h = spot[0]
            self._callout(spot, note, at or (x + w / 2, 905))
        return self._rgb()

    STRIP = ((32, 762, 916, 306), (972, 762, 916, 306))  # the close-up's two small charts

    @staticmethod
    def side_box(k):
        """Pixel box of side k (map and panels) in the close-up."""
        return 18 + 940 * k, 96, 922, 660

    @staticmethod
    def tags(frame, view, t, ids, color, size=17, lw=2.5):
        """Each (order id, label) in ids drawn by its pickup place once released: hollow while
        waiting, filled once picked up."""
        nodes, at = view.r["map"]["nodes"], {}  # tags already drawn per place, to stack them
        for oid, tag in ids:
            o = view.orders[oid]
            if o["release"] > t:
                continue
            n, got = nodes[o["from"]], o["pickup"] is not None and o["pickup"] <= t
            m = at[o["from"]] = at.get(o["from"], -1) + 1
            xy = (n["x"] - 0.32 - 0.24 * m, n["y"] - 0.3)
            frame.add_patch(Circle(xy, 0.11, fc=color if got else PANEL, ec=color, lw=lw,
                                   zorder=6))
            frame.text(*xy, tag, ha="center", va="center", fontsize=pt(size),
                       color="white" if got else color, weight="bold", zorder=7)

    @staticmethod
    def trail(frame, r, since, t, color, scale=1.0):
        """The route flown between since and t: solid with boxes on board, dashed empty."""
        for g in r["segments"]:
            if g[0] != "fly" or g[3] == g[4] or g[2] <= since or g[1] >= t:
                continue
            path = np.asarray(r["map"]["routes"][f"{g[3]}-{g[4]}"], dtype=float)
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            a, b = (np.clip((v - g[1]) / max(g[2] - g[1], EPS), 0, 1) * cum[-1] for v in (since, t))
            xs = np.r_[a, cum[(cum > a) & (cum < b)], b]
            pts = np.column_stack([np.interp(xs, cum, path[:, 0]), np.interp(xs, cum, path[:, 1])])
            loaded = bool(state_at(r, g[1] + EPS)["cargo"])
            frame.plot(pts[:, 0], pts[:, 1], color=color, lw=(5 if loaded else 3) * scale, alpha=0.6,
                       ls="-" if loaded else (0, (2, 1.5)), solid_capstyle="round", zorder=1.5)

    # what each step of the screen guide lights up: pixel boxes, then its subtitle
    GUIDE = [([(26, 182, 572, 572), (970, 182, 572, 572)], "The map", (784, 900)),
             ([(602, 182, 348, 572), (1546, 182, 348, 572)], "Each drone", (1258, 900)),
             ([(26, 800, 1868, 274)], "Running totals", (960, 520))]

    def guide(self, t: float, step: int, note: str) -> np.ndarray:
        """The frame at t, dimmed except the boxes of one guide step, with a callout holding
        note (lines split at newlines) and an arrow to each lit box."""
        self.draw(t)
        boxes, _, centre = self.GUIDE[step]
        spotlight(self.fig, (self.W, self.H), boxes, 1.0)
        self._callout(boxes, note, centre)
        return self._rgb()

    def _callout(self, boxes, note, centre):
        """A white box holding note (lines split at newlines) at centre, with an arrow to each
        of boxes."""
        cx, cy = centre
        over = self.fig.add_axes((0, 0, 1, 1), zorder=11)
        over.set(xlim=(0, self.W), ylim=(self.H, 0))
        over.axis("off")
        lines = note.split("\n")
        w, h = 22 * max(map(len, lines)) + 60, 52 * len(lines) + 40
        for x, y, bw, bh in boxes:  # from the callout's edge to the nearest point of the box
            tx, ty = min(max(cx, x + 20), x + bw - 20), min(max(cy, y + 20), y + bh - 20)
            over.annotate("", (tx, ty), (cx, cy), zorder=11, arrowprops=dict(
                arrowstyle="-|>", color="#e8590c", lw=4, mutation_scale=34, shrinkA=h / 2 * 0.9))
        over.add_patch(rounded(cx - w / 2, cy - h / 2, w, h, 18, fc="white", ec="#e8590c", lw=4,
                               zorder=12))
        for j, line in enumerate(lines):
            over.text(cx, cy - h / 2 + 24 + 52 * j, line, fontsize=pt(34), color=INK,
                      weight="bold", ha="center", va="top", zorder=13)

    def results(self, title, rows, note, takeaway="", hot=()) -> np.ndarray:
        """A bar chart slide; rows = (label, colour, medium %, high %, ms per decision[, line]);
        the rows labelled in hot stand out (a band, their bars and times), the others fade."""
        ax = self._canvas()
        self._page(ax)
        lit = [not hot or row[0] in hot for row in rows]
        band = 700 / len(rows)  # row height in pixels, as below
        for i, on in enumerate(lit):
            if hot and on:
                ax.add_patch(rounded(100, 280 + band * i + 4, 1720, band - 8, 12, fc="#fff4e6",
                                     ec="#e8590c", lw=2))
        self._text(ax, 120, 90, title, 44, INK, "bold")
        for k, part in enumerate(wrap(note, 112)):
            self._text(ax, 120, 152 + 32 * k, part, 24, MUTED)
        top, tall = 280, 700
        for j, load in enumerate(("Medium demand", "High demand")):
            c = self._box(470 + 560 * j, top, 510, tall)
            vals = [row[2 + j] for row in rows]
            ys = np.arange(len(rows))[::-1]
            for yy, v, row, on in zip(ys, vals, rows, lit):
                c.barh(yy, v, color=row[1], height=0.64, alpha=1.0 if on else 0.25)
                c.text(v - 1, yy, f"{v:.1f}%", ha="right", va="center", fontsize=pt(21),
                       color="white", weight="bold", alpha=1.0 if on else 0.6)
            c.patch.set_alpha(0)
            c.set(xlim=(0, 100), ylim=(-0.6, len(rows) - 0.4))
            c.set_yticks([])
            c.set_xticks([0, 50, 100], ["0", "50", "100 %"])
            c.tick_params(labelsize=pt(17), colors=MUTED, length=0)
            for side in ("top", "right", "left"):
                c.spines[side].set_visible(False)
            c.spines["bottom"].set_color(LINE)
            c.set_title(f"{load} · on time", fontsize=pt(26), color=INK, weight="bold",
                        loc="left", pad=16)
        step = tall / len(rows)
        for i, (row, on) in enumerate(zip(rows, lit)):
            yy = top + step * (i + 0.5) - 18
            strong = on and (bool(hot) or i < 2)  # without hot: PPO and the horizon
            self._text(ax, 120, yy - 12, row[0], 26, INK if on else FAINT,
                       "bold" if strong else "normal")
            if len(row) > 5:
                self._text(ax, 120, yy + 20, row[5], 19, MUTED if on else FAINT)
            ax.add_patch(rounded(1600, yy - 10, 200, 52, 26, fc=row[1] if strong else WASH,
                                 ec="none"))
            self._text(ax, 1700, yy - 2, row[4], 26, "white" if strong else FAINT,
                       "bold" if strong else "normal", ha="center")
        self._text(ax, 1700, 244, "TIME PER DECISION", 20, INK, "bold", ha="center")
        for k, part in enumerate(wrap(takeaway, 90)):
            self._text(ax, 120, 1000 + 38 * k, part, 28, INK, "bold")
        return self._rgb()

    def ways(self, items, lit=None) -> np.ndarray:
        """Why PPO is more efficient: it flies less empty, in the ways of items (name, line);
        with lit, that way stands out and the others fade, as on an agenda."""
        ax = self._canvas()
        self._page(ax)
        self._text(ax, self.W / 2, 150, "Why PPO is more efficient", 44, MUTED, "bold",
                   ha="center")
        self._text(ax, self.W / 2, 230, "It flies less empty than Slack-aware", 72, self.colors[0],
                   "bold", ha="center")
        for k, (name, line) in enumerate(items):
            x, on = 216 + 780 * k, lit is None or k == lit
            self._panel(ax, x, 470, 708, 300, edgecolor=self.colors[0] if on else LINE,
                        lw=6 if lit == k else 4, facecolor="#fff4e6" if lit == k else PANEL)
            self._text(ax, x + 354, 520, name, 44, INK if on else FAINT, "bold", ha="center")
            for j, part in enumerate(wrap(line, 30)):
                self._text(ax, x + 354, 620 + 54 * j, part, 36, INK if on else FAINT, ha="center")
        return self._rgb()

    def versus(self, columns, foot) -> np.ndarray:
        """Two dispatchers facing each other: per column (name, one line, badge or ""), a large
        VS between."""
        ax = self._canvas()
        self._page(ax)
        for k, (name, line, badge) in enumerate(columns):
            x, color = 96 + 1000 * k, self.colors[k]
            self._panel(ax, x, 280, 728, 420, edgecolor=color, lw=4)
            self._text(ax, x + 364, 330, name, 52, color, "bold", ha="center")
            parts = wrap(line, 34)
            for j, part in enumerate(parts):
                self._text(ax, x + 364, 430 + 54 * j, part, 36, INK, ha="center")
            if badge:
                bw, by = 19 * len(badge) + 48, 450 + 54 * len(parts)
                ax.add_patch(rounded(x + 364 - bw / 2, by, bw, 52, 26, fc=color, ec="none"))
                self._text(ax, x + 364, by + 8, badge, 28, "white", "bold", ha="center")
        self._text(ax, self.W / 2, 420, "VS", 96, INK, "bold", ha="center")
        if foot:
            self._text(ax, self.W / 2, 760, foot, 34, MUTED, ha="center")
        return self._rgb()

    def day_numbers(self) -> np.ndarray:
        """The whole day for both: final outcomes, battery over the day, and loaded and empty
        kilometres."""
        ax = self._canvas()
        self._page(ax)
        self._text(ax, 96, 96, "This day in numbers", 52, INK, "bold")
        peaks, hours = self.r["meta"]["peaks"], np.arange(0, self.end + 1, 120)
        boxes = [(96, 230, 560, 640), (700, 230, 560, 640), (1304, 230, 520, 640)]
        titles = ["Results", "Battery: PPO tops up often", "Distance flown (km)"]
        for (x, y, w, h), title in zip(boxes, titles):
            self._panel(ax, x, y, w, h)
            self._text(ax, x + 24, y + 18, title, 24, INK, "bold")
        style = lambda c: (c.set_xlim(0, self.end), c.set_xticks(hours, [clock(v) for v in hours]),
                           c.tick_params(labelsize=pt(16), colors=MUTED, length=0),
                           [c.spines[side].set_visible(False) for side in ("top", "right")],
                           c.grid(axis="y", color=WASH, lw=1), c.set_facecolor(PANEL),
                           [c.axvspan(a, b, color=WASH, lw=0) for a, b in peaks])
        x, y, w, h = boxes[0]
        c = self._box(x + 40, y + 80, w - 70, h - 150)
        top = max(v.r["meta"]["summary"]["orders"] for v in self.views)
        for k, v in enumerate(self.views):
            sm = v.r["meta"]["summary"]
            vals = (sm["on_time"], sm["delivered"] - sm["on_time"], sm["expired"])
            xs = np.arange(3) + (k - 0.5) * 0.38
            c.bar(xs, vals, width=0.36, color=self.colors[k], label=self.titles[k])
            for xx, val in zip(xs, vals):
                c.text(xx, val + top * 0.02, str(val), ha="center", va="bottom", fontsize=pt(18),
                       color=INK, weight="bold")
        c.set_xticks(range(3), ("on time", "late", "expired"))
        c.set(ylim=(0, top * 0.95))
        c.set_yticks([])
        c.tick_params(labelsize=pt(18), colors=INK, length=0)
        for side in ("top", "right", "left"):
            c.spines[side].set_visible(False)
        c.set_facecolor(PANEL)
        c.legend(loc="upper right", frameon=False, fontsize=pt(17))
        x, y, w, h = boxes[1]
        for k, v in enumerate(self.views):  # one row per dispatcher
            c = self._box(x + 70, y + 80 + 270 * k, w - 100, 210)
            style(c)
            g = v.r["segments"]
            c.fill_between([u for s_ in g for u in (s_[1], s_[2])],
                           [u for s_ in g for u in (s_[5], s_[6])], color=self.colors[k],
                           alpha=0.25, lw=0)
            c.plot([u for s_ in g for u in (s_[1], s_[2])], [u for s_ in g for u in (s_[5], s_[6])],
                   color=self.colors[k], lw=2)
            c.set_ylim(0, 105)
            c.set_yticks([0, 50, 100])
            c.set_title(self.titles[k], loc="left", fontsize=pt(17), color=self.colors[k],
                        weight="bold")
        x, y, w, h = boxes[2]
        c = self._box(x + 40, y + 80, w - 70, h - 150)
        for k, v in enumerate(self.views):
            loaded = empty = 0.0
            for g in v.r["segments"]:
                if g[0] == "fly" and g[3] != g[4]:
                    path = np.asarray(v.r["map"]["routes"][f"{g[3]}-{g[4]}"])
                    km = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                    if state_at(v.r, g[1] + EPS)["cargo"]:
                        loaded += km
                    else:
                        empty += km
            c.bar(k, loaded, color=self.colors[k], width=0.6)
            c.bar(k, empty, bottom=loaded, color=self.colors[k], alpha=0.35, width=0.6,
                  hatch="//", ec="white")
            c.text(k, loaded / 2, f"{loaded:.0f}\nwith boxes", ha="center", va="center",
                   fontsize=pt(16), color="white", weight="bold")
            c.text(k, loaded + empty / 2, f"{empty:.0f}\nempty", ha="center", va="center",
                   fontsize=pt(16), color=INK, weight="bold")
        c.set_xticks([0, 1], [t.split()[0] for t in self.titles])
        c.tick_params(labelsize=pt(18), colors=INK, length=0)
        c.set_yticks([])
        for side in ("top", "right", "left"):
            c.spines[side].set_visible(False)
        c.set_facecolor(PANEL)
        return self._rgb()

    def bullets(self, title, lines, kicker, highlight="", size=36) -> np.ndarray:
        """A heading with numbered points, and optionally a highlighted box under them."""
        ax = self._canvas()
        self._page(ax)
        self._text(ax, 120, 110, kicker, 22, MUTED, "bold")
        self._text(ax, 120, 150, title, 58, INK, "bold")
        width, step = int(1560 / (size * 0.54)), round(size * 1.42)
        y = 290
        for j, line in enumerate(lines):
            ax.add_patch(Circle((150, y + size * 0.72), size * 0.78, fc=self.colors[0]))
            ax.text(150, y + size * 0.72, str(j + 1), ha="center", va="center",
                    fontsize=pt(size * 0.75), color="white", weight="bold")
            parts = wrap(line, width)
            for k, part in enumerate(parts):
                self._text(ax, 208, y + step * k, part, size, INK)
            y += step * len(parts) + round(size * 1.15)
        if highlight:
            parts = wrap(highlight, int(1580 / (size * 0.56)))
            ax.add_patch(rounded(120, y + 4, self.W - 240, step * len(parts) + 44, 16,
                                 fc="#fff4e6", ec="#e8590c", lw=3))
            for k, part in enumerate(parts):
                self._text(ax, 150, y + 26 + step * k, part, size, "#b34700", "bold")
        return self._rgb()

    def closing(self, title, lines, size=104) -> np.ndarray:
        """A centred title and a few lines: the first and the last slide."""
        ax = self._canvas()
        self._page(ax)
        ax.text(self.W / 2, 440, title, ha="center", va="center", fontsize=pt(size), color=INK,
                weight="bold")
        for j, line in enumerate(lines):
            ax.text(self.W / 2, 610 + 80 * j, line, ha="center", va="center", fontsize=pt(44),
                    color=MUTED if j else INK)
        return self._rgb()

    def card(self, title, lines, kicker) -> np.ndarray:
        """A full-screen text slide."""
        ax = self._canvas()
        self._page(ax)
        self._text(ax, 120, 150, kicker, 26, MUTED, "bold")
        self._text(ax, 120, 200, title, 62, INK, "bold")
        for j, line in enumerate(lines):
            self._text(ax, 120, 380 + 88 * j, line, 44)
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
