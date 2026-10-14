"""Matplotlib rendering of the map, drone and orders at any time up to the current one."""
import math
import os
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from medical_uav.env.config import (CHARGERS, EPS, FULL_BATTERY, M, NODE_NAMES, ORDER_TYPES,
                                    EnvConfig)
from medical_uav.env.map_gen import CityMap

TYPE_COLORS = {"plasma": "#d62728", "medicine": "#1f77b4", "sample": "#9467bd",
               "supplies": "#ff7f0e"}
DAY_START_HOUR = 8  # clock hour at simulation time 0
PANEL_ROWS = 8  # pending orders listed in the side panel
FRAME_PX = (960, 528)  # frame width and height in pixels; video codecs need multiples of 16
DPI = 80

TEXT, MUTED, AMBER, RED = "#222222", "#8a8a8a", "#c77c00", "#d62728"
NO_FLY, CHARGER_RING, ROUTE, BATTERY_FILL = "#f4c4c4", "#2ca02c", "#555555", "#5b9b5f"
DUE_WARNING_MIN = 10.0  # due times below this many minutes are shown in amber


def _color(o) -> str:
    return TYPE_COLORS[ORDER_TYPES[o.type_idx].name]


def clock(t: float) -> str:
    """Return simulation time t (minutes) as a wall-clock time, HH:MM."""
    minutes = int(t)
    return f"{DAY_START_HOUR + minutes // 60:02d}:{minutes % 60:02d}"


def node_name(node: int) -> str:
    """Return the display name of a node; the charger at M is shown as M (C1)."""
    return "M (C1)" if node == M else NODE_NAMES[node]


def _point_along(path: np.ndarray, fraction: float) -> np.ndarray:
    """Return the point at the given fraction of the polyline's length."""
    if len(path) == 1:
        return path[0]
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    target = fraction * seg_len.sum()  # distance still to go along the path
    for i, length in enumerate(seg_len):
        if target <= length or i == len(seg_len) - 1:
            f = 0.0 if length == 0 else min(target / length, 1.0)  # fraction along this segment
            return path[i] + f * (path[i + 1] - path[i])
        target -= length
    return path[-1]


def _leave_time(o):
    """Return when box o left the drone, or None if it is still on board."""
    return o.end_time if o.status == "delivered" else o.disposed_time


def drone_state_at(sim, t: float) -> dict:
    """Reconstruct the drone position, battery, cargo and pending orders at time t."""
    seg = None
    for s in sim.segments:
        if s.t0 - EPS <= t <= s.t1 + EPS:
            seg = s
            if t < s.t1 - EPS:
                break
    if seg is None:
        xy, battery, activity = sim.city_map.node_xy[sim.location], sim.battery, "idle"
    else:
        span = seg.t1 - seg.t0
        # fraction of the current activity completed at t
        f = 1.0 if span <= EPS else min(max((t - seg.t0) / span, 0.0), 1.0)
        if seg.kind == "fly":
            xy = _point_along(sim.city_map.paths[(seg.node_from, seg.node_to)], f)
        else:
            xy = sim.city_map.node_xy[seg.node_to]
        battery = seg.battery0 + f * (seg.battery1 - seg.battery0)
        activity = seg.kind

    cargo = [o for o in sim.orders
             if o.pickup_time is not None and o.pickup_time <= t + EPS
             and (_leave_time(o) is None or _leave_time(o) > t + EPS)]
    pending = [o for o in sim.orders
               if o.release <= t + EPS and o.status != "rejected"
               and (o.pickup_time is None or o.pickup_time > t + EPS)
               and (o.end_time is None or o.end_time > t + EPS)]
    return {"xy": np.asarray(xy, dtype=float), "battery": battery, "activity": activity,
            "cargo": cargo, "pending": pending, "segment": seg}


def activity_text(state: dict, orders_by_id: dict, location: int) -> str:
    """Return one line describing what the drone is doing in this state."""
    seg = state["segment"]
    if seg is None:
        return f"At {node_name(location)}"
    here = NODE_NAMES[seg.node_to]
    o = orders_by_id.get(seg.order_id)
    label = f"#{o.id} {ORDER_TYPES[o.type_idx].name}" if o is not None else ""
    if seg.kind == "fly":
        return f"Flying {NODE_NAMES[seg.node_from]} \u2192 {here}"
    if seg.kind == "load":
        return f"Loading {label} at {here}, for {NODE_NAMES[o.dest]}"
    if seg.kind == "unload":
        return f"Unloading {label} at {here}"
    if seg.kind == "dispose":
        return f"Unloading expired {label} at {here}"
    if seg.kind == "charge":
        return (f"Charging at {node_name(seg.node_to)}: "
                f"{seg.battery0:.0f}% \u2192 {seg.battery1:.0f}%")
    return f"Waiting at {here}"


def outcome_counts(sim, t: float) -> dict:
    """Count released, delivered, on-time, expired, rejected and open orders at time t."""
    def closed(o):
        return o.end_time is not None and o.end_time <= t + EPS

    released = [o for o in sim.orders if o.status != "future" and o.release <= t + EPS]
    delivered = [o for o in released if o.status == "delivered" and closed(o)]
    expired = sum(1 for o in released if o.status == "expired" and closed(o))
    rejected = sum(1 for o in released if o.status == "rejected" and closed(o))
    return {"released": len(released), "delivered": len(delivered),
            "on_time": sum(1 for o in delivered if o.end_time - o.release <= o.limit + EPS),
            "expired": expired, "rejected": rejected,
            "open": len(released) - len(delivered) - expired - rejected}


def _due_color(minutes: float) -> str:
    if minutes < 0:
        return RED
    return AMBER if minutes < DUE_WARNING_MIN else TEXT


def _fraction_box(x: float, y: float, w: float, h: float) -> list[float]:
    """Convert a pixel box (x, y from the top-left corner) to matplotlib figure fractions."""
    fw, fh = FRAME_PX
    return [x / fw, 1 - (y + h) / fh, w / fw, h / fh]


def _pixel_axes(fig, x: float, y: float, w: float, h: float):
    """Add axes at a pixel box, drawn in pixel units with y pointing down."""
    ax = fig.add_axes(_fraction_box(x, y, w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


class Renderer:
    """Draws 960 x 528 frames: header, map, status panel and a legend footer."""

    def __init__(self, city_map: CityMap, cfg: EnvConfig, title: str = "Medical UAV"):
        self.city_map = city_map
        self.cfg = cfg
        self.title = title
        self.fig = Figure(figsize=(FRAME_PX[0] / DPI, FRAME_PX[1] / DPI), dpi=DPI)
        FigureCanvasAgg(self.fig)
        self.boxes = {"head": (16, 6, 928, 44), "map": (16, 58, 404, 404),
                      "panel": (448, 58, 496, 404), "foot": (16, 472, 928, 46)}

    def draw(self, sim, t: float, decision: str = "") -> np.ndarray:
        """Return an RGB image of the day at time t; `decision` is shown under the title."""
        self.fig.clear()
        state = drone_state_at(sim, t)
        orders_by_id = {o.id: o for o in sim.orders}
        self._draw_header(t, decision)
        self._draw_map(state)
        self._draw_panel(sim, t, state, orders_by_id)
        self._draw_footer(sim, t)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _draw_header(self, t: float, decision: str) -> None:
        ax = _pixel_axes(self.fig, *self.boxes["head"])
        w = self.boxes["head"][2]
        ax.text(0, 14, self.title, fontsize=13, weight="bold", color=TEXT, va="center")
        if decision:
            ax.text(0, 34, decision, fontsize=10.5, color=MUTED, va="center")
        ax.text(w, 14, clock(t), fontsize=16, weight="bold", color=TEXT, ha="right", va="center")
        ax.text(w, 34, f"t = {t:.1f} min", fontsize=10.5, color=MUTED, ha="right", va="center")

    def _draw_map(self, state: dict) -> None:
        cm, cfg = self.city_map, self.cfg
        ax = self.fig.add_axes(_fraction_box(*self.boxes["map"]))
        size = cfg.grid_size * cfg.cell_km
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for row, col, h, w in cm.zones:
            ax.add_patch(Rectangle((col * cfg.cell_km, row * cfg.cell_km), w * cfg.cell_km,
                                   h * cfg.cell_km, color=NO_FLY, lw=0))

        seg = state["segment"]
        if seg is not None and seg.kind == "fly":
            path = cm.paths[(seg.node_from, seg.node_to)]
            ax.plot(path[:, 0], path[:, 1], "--", color=ROUTE, lw=1.5)

        for i, (x, y) in enumerate(cm.node_xy):
            if i in CHARGERS:
                ax.add_patch(Circle((x, y), 0.17, fill=False, color=CHARGER_RING, lw=2))
            ax.plot(x, y, "s" if i == M else "o", color="#333333", ms=9)
            label = "M/C1" if i == M else NODE_NAMES[i]
            above = y > 0.5  # label above the node unless it is near the top edge
            ax.text(x, y - 0.22 if above else y + 0.22, label, ha="center",
                    va="bottom" if above else "top", fontsize=10, weight="bold")

        dx, dy = state["xy"]
        ax.plot(dx, dy, marker="P", color="#111111", ms=14, mec="white", mew=1.5)
        for k, o in enumerate(state["cargo"]):
            ax.plot(dx + 0.12 * (k - 1), dy + 0.25, "s", color=_color(o), ms=6)

        # scale bar and map size, bottom left
        x0, y0 = 0.15, size - 0.18
        ax.plot([x0, x0 + 1.0], [y0, y0], color=TEXT, lw=2)
        ax.text(x0 + 0.5, y0 - 0.08, "1 km", ha="center", va="bottom", fontsize=9, color=TEXT)
        ax.text(size - 0.1, size - 0.1, f"{size:g} x {size:g} km", ha="right", va="bottom",
                fontsize=9, color=MUTED)

    def _draw_panel(self, sim, t: float, state: dict, orders_by_id: dict) -> None:
        ax = _pixel_axes(self.fig, *self.boxes["panel"])
        cfg = self.cfg
        due_x, exp_x = 396, 490

        ax.text(0, 12, activity_text(state, orders_by_id, sim.location), fontsize=12.5,
                weight="bold", color=TEXT, va="center")

        battery = state["battery"]
        ax.text(0, 46, f"Battery {battery:5.1f}%", fontsize=11, color=TEXT, va="center")
        bar_x, bar_w = 130, 300
        ax.add_patch(Rectangle((bar_x, 39), bar_w, 14, fill=False, lw=1, color=TEXT))
        ax.add_patch(Rectangle((bar_x, 39), bar_w * battery / FULL_BATTERY, 14, color=BATTERY_FILL))
        reserve_x = bar_x + bar_w * cfg.reserve / FULL_BATTERY
        ax.plot([reserve_x, reserve_x], [35, 57], color=TEXT, lw=1.2)
        ax.text(reserve_x + 4, 64, f"reserve {cfg.reserve:g}%", fontsize=8.5, color=MUTED,
                va="center")
        load = sum(o.weight for o in state["cargo"])
        ax.text(0, 72, f"Cargo {len(state['cargo'])}/{cfg.max_boxes} boxes, "
                       f"{load:.2f}/{cfg.max_weight:.2f} kg", fontsize=11, color=TEXT, va="center")

        y = 100
        self._table_header(ax, y, "On board", due_x, exp_x)
        cargo = sorted(state["cargo"], key=lambda o: (o.deadline, o.id))
        for k in range(cfg.max_boxes):
            y += 19
            if k >= len(cargo):
                ax.text(16, y, "empty", fontsize=10, color=MUTED, va="center")
                continue
            o = cargo[k]
            self._order_row(ax, y, o, f"\u2192 {NODE_NAMES[o.dest]}", t, due_x, exp_x,
                            expired=o.status == "expired" and o.expiry <= t + EPS)

        y += 30
        pending = sorted(state["pending"], key=lambda o: (o.deadline, o.id))
        self._table_header(ax, y, f"Pending {len(pending)} (earliest deadline first)",
                           due_x, exp_x)
        shown = pending if len(pending) <= PANEL_ROWS else pending[:PANEL_ROWS - 1]
        for o in shown:
            y += 18
            route = f"{NODE_NAMES[o.origin]} \u2192 {NODE_NAMES[o.dest]}"
            self._order_row(ax, y, o, route, t, due_x, exp_x)
        if len(pending) > len(shown):
            y += 18
            ax.text(16, y, f"+{len(pending) - len(shown)} more", fontsize=10, color=MUTED,
                    va="center")
        ax.text(0, 398, "due < 0: late, still deliverable until it expires", fontsize=8.5,
                color=MUTED, va="center")

    @staticmethod
    def _table_header(ax, y: float, title: str, due_x: float, exp_x: float) -> None:
        ax.text(0, y, title, fontsize=11, weight="bold", color=TEXT, va="center")
        ax.text(due_x, y, "due (min)", fontsize=9, color=MUTED, ha="right", va="center")
        ax.text(exp_x, y, "expires (min)", fontsize=9, color=MUTED, ha="right", va="center")

    @staticmethod
    def _order_row(ax, y: float, o, where: str, t: float, due_x: float, exp_x: float,
                   expired: bool = False) -> None:
        ax.add_patch(Rectangle((2, y - 5), 9, 9, color=_color(o)))
        ax.text(16, y, f"#{o.id}", fontsize=10, color=MUTED, va="center")
        ax.text(56, y, ORDER_TYPES[o.type_idx].name, fontsize=10, color=TEXT, va="center")
        ax.text(130, y, where, fontsize=10, color=TEXT, va="center")
        if expired:
            ax.text(due_x, y, "EXPIRED", fontsize=10, color=RED, weight="bold", ha="right",
                    va="center")
            return
        due = o.deadline - t
        ax.text(due_x, y, f"{due:+.0f}", fontsize=10, color=_due_color(due), ha="right",
                va="center")
        ax.text(exp_x, y, f"{o.expiry - t:.0f}", fontsize=10, color=TEXT, ha="right",
                va="center")

    def _draw_footer(self, sim, t: float) -> None:
        ax = _pixel_axes(self.fig, *self.boxes["foot"])
        w = self.boxes["foot"][2]
        ax.plot([0, w], [1, 1], color="#dddddd", lw=1)
        x = 0
        for draw, label in [(lambda x: ax.plot(x, 15, "P", color="#111111", ms=10), "drone"),
                            (lambda x: ax.plot([x - 7, x + 7], [15, 15], "--", color=ROUTE),
                             "current flight"),
                            (lambda x: ax.plot(x, 15, "s", color="#333333", ms=7),
                             "medical center"),
                            (lambda x: ax.plot(x, 15, "o", color="#333333", ms=7), "hospital"),
                            (lambda x: ax.plot(x, 15, "o", mfc="none", mec=CHARGER_RING, ms=11,
                                               mew=2), "charger"),
                            (lambda x: ax.add_patch(Rectangle((x - 7, 9), 14, 12, color=NO_FLY)),
                             "no-fly zone")]:
            draw(x + 8)
            ax.text(x + 20, 15, label, fontsize=9.5, color=TEXT, va="center")
            x += 30 + 7 * len(label)
        x = 0
        for order_type in ORDER_TYPES:
            ax.add_patch(Rectangle((x + 3, 30), 10, 10, color=TYPE_COLORS[order_type.name]))
            ax.text(x + 20, 35, order_type.name, fontsize=9.5, color=TEXT, va="center")
            x += 34 + 7 * len(order_type.name)

        c = outcome_counts(sim, t)
        late = c["delivered"] - c["on_time"]
        ax.text(w, 15, f"Delivered {c['delivered']} (on time {c['on_time']}, late {late})",
                fontsize=10.5, color=TEXT, ha="right", va="center")
        ax.text(w, 35, f"Expired {c['expired']}   Rejected {c['rejected']}   Open {c['open']}",
                fontsize=10.5, color=TEXT, ha="right", va="center")


def frame_times(t0: float, t1: float, step: float) -> list[float]:
    """Return frame times after t0 up to t1, every `step` minutes, always including t1."""
    times = []
    k = math.floor(t0 / step + EPS) + 1
    while k * step <= t1 + EPS:
        times.append(k * step)
        k += 1
    if t1 > t0 + EPS and (not times or times[-1] < t1 - EPS):
        times.append(t1)
    return times


class GifWriter:
    """Write frames to a temporary MP4 and convert it to a GIF on close, keeping memory low."""

    def __init__(self, path: str, fps: int):
        self.path = path
        self.tmp_path = path + ".tmp.mp4"
        self.writer = imageio.get_writer(self.tmp_path, fps=fps)

    def append_data(self, frame) -> None:
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()
        # build a colour palette from the video, then apply it to the GIF
        palette = "split[a][b];[a]palettegen[p];[b][p]paletteuse"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", self.tmp_path,
               "-vf", palette, "-loop", "0", self.path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            os.remove(self.tmp_path)


def open_animation_writer(path: str, fps: int):
    """Return a frame writer for an .mp4 or .gif path."""
    if path.endswith(".mp4"):
        return imageio.get_writer(path, fps=fps)
    if path.endswith(".gif"):
        return GifWriter(path, fps)
    raise ValueError(f"animation must be .gif or .mp4, got {path!r}")


def save_animation(frames, path: str, fps: int) -> int:
    """Write frames one at a time and return how many were written."""
    writer = open_animation_writer(path, fps)
    count = 0
    try:
        for frame in frames:
            writer.append_data(frame)
            count += 1
    finally:
        writer.close()
    return count
