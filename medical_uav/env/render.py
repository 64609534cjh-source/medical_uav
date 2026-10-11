"""Matplotlib rendering. Frames are rebuilt from the simulator history, so any
time t <= sim.t can be drawn (used for smooth replays and videos)."""
import math

import imageio.v2 as imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from medical_uav.env.config import CHARGERS, EPS, NODE_NAMES, ORDER_TYPES, EnvConfig
from medical_uav.env.map_gen import CityMap

TYPE_COLORS = ("#d62728", "#1f77b4", "#9467bd", "#ff7f0e")  # plasma, medicine, sample, supplies
DAY_START_HOUR = 8


def _point_along(path: np.ndarray, fraction: float) -> np.ndarray:
    """Point at the given fraction of the polyline's arc length."""
    if len(path) == 1:
        return path[0]
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    target = fraction * seg_len.sum()
    for i, length in enumerate(seg_len):
        if target <= length or i == len(seg_len) - 1:
            f = 0.0 if length == 0 else min(target / length, 1.0)
            return path[i] + f * (path[i + 1] - path[i])
        target -= length
    return path[-1]


def _leave_time(o):
    return o.end_time if o.status == "delivered" else o.disposed_time


def drone_state_at(sim, t: float) -> dict:
    """Drone position, battery, cargo and pending orders at time t (from the history)."""
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


class Renderer:
    def __init__(self, city_map: CityMap, cfg: EnvConfig):
        self.city_map = city_map
        self.cfg = cfg
        self.fig = Figure(figsize=(12, 6.5), dpi=80)
        FigureCanvasAgg(self.fig)
        self.ax_map = self.fig.add_axes([0.03, 0.05, 0.55, 0.9])
        self.ax_panel = self.fig.add_axes([0.61, 0.05, 0.37, 0.9])

    def draw(self, sim, t: float) -> np.ndarray:
        state = drone_state_at(sim, t)
        self._draw_map(sim, state)
        self._draw_panel(sim, t, state)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _draw_map(self, sim, state) -> None:
        ax, cm, cfg = self.ax_map, self.city_map, self.cfg
        ax.clear()
        size = cfg.grid_size * cfg.cell_km
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for row, col, h, w in cm.zones:
            ax.add_patch(Rectangle((col * cfg.cell_km, row * cfg.cell_km), w * cfg.cell_km,
                                   h * cfg.cell_km, color="#f4b6b6", alpha=0.8, lw=0))

        seg = state["segment"]
        if seg is not None and seg.kind == "fly":
            path = cm.paths[(seg.node_from, seg.node_to)]
            ax.plot(path[:, 0], path[:, 1], "--", color="#555555", lw=1.5)

        for i, (x, y) in enumerate(cm.node_xy):
            if i in CHARGERS:
                ax.add_patch(Circle((x, y), 0.17, fill=False, color="#2ca02c", lw=2))
            ax.plot(x, y, "s" if i == 0 else "o", color="#333333", ms=9)
            label = "M/C1" if i == 0 else NODE_NAMES[i]
            above = y > 0.5  # near the top edge, put the label below the node
            ax.text(x, y - 0.22 if above else y + 0.22, label, ha="center",
                    va="bottom" if above else "top", fontsize=10, weight="bold")

        dx, dy = state["xy"]
        ax.plot(dx, dy, marker="P", color="#111111", ms=14, mec="white", mew=1.5)
        for k, o in enumerate(state["cargo"]):
            ax.plot(dx + 0.12 * (k - 1), dy + 0.25, "s", color=TYPE_COLORS[o.type_idx], ms=6)

    def _draw_panel(self, sim, t: float, state) -> None:
        ax = self.ax_panel
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        minutes = int(t)
        clock = f"{DAY_START_HOUR + minutes // 60:02d}:{minutes % 60:02d}"
        ax.text(0, 0.97, f"{clock}  (t = {t:5.1f} min)", fontsize=14, weight="bold")
        ax.text(0, 0.92, f"activity: {state['activity']}", fontsize=11)

        battery = state["battery"]
        ax.text(0, 0.86, f"battery {battery:5.1f}%", fontsize=11)
        ax.add_patch(Rectangle((0.35, 0.855), 0.6, 0.03, fill=False, lw=1))
        color = "#2ca02c" if battery > 40 else ("#ff7f0e" if battery > 25 else "#d62728")
        ax.add_patch(Rectangle((0.35, 0.855), 0.6 * battery / 100, 0.03, color=color))
        ax.plot([0.35 + 0.6 * self.cfg.reserve / 100] * 2, [0.85, 0.89], color="k", lw=1)

        ax.text(0, 0.79, "cargo", fontsize=11, weight="bold")
        for k in range(self.cfg.max_boxes):
            y = 0.74 - 0.05 * k
            if k < len(state["cargo"]):
                o = state["cargo"][k]
                left = o.deadline - t
                text = (f"{ORDER_TYPES[o.type_idx].name:8s} -> {NODE_NAMES[o.dest]}  "
                        f"{o.weight:.2f} kg  {left:+5.0f} min")
                ax.text(0.02, y, text, fontsize=10, family="monospace",
                        color=TYPE_COLORS[o.type_idx])
            else:
                ax.text(0.02, y, "[ empty ]", fontsize=10, family="monospace", color="#999999")

        pending = sorted(state["pending"], key=lambda o: o.deadline)
        ax.text(0, 0.57, f"pending orders ({len(pending)})", fontsize=11, weight="bold")
        for k, o in enumerate(pending[:12]):
            left = o.deadline - t
            color = "#2ca02c" if left > 10 else ("#ff7f0e" if left > 0 else "#d62728")
            text = (f"{ORDER_TYPES[o.type_idx].name:8s} {NODE_NAMES[o.origin]:>2s}->"
                    f"{NODE_NAMES[o.dest]:2s} {left:+5.0f} min")
            ax.text(0.02, 0.52 - 0.04 * k, text, fontsize=9.5, family="monospace", color=color)

        done = [o for o in sim.orders if o.status == "delivered" and o.end_time <= t + EPS]
        on_time = sum(1 for o in done if o.end_time - o.release <= o.limit + EPS)
        lost = sum(1 for o in sim.orders
                   if o.status in ("expired", "rejected") and o.end_time <= t + EPS)
        ax.text(0, 0.0, f"delivered {len(done)} (on time {on_time})   lost {lost}", fontsize=11)


def frame_times(t0: float, t1: float, step: float) -> list[float]:
    """Frame times in (t0, t1] on a grid of `step` minutes, always ending at t1."""
    times = []
    k = math.floor(t0 / step + EPS) + 1
    while k * step <= t1 + EPS:
        times.append(k * step)
        k += 1
    if t1 > t0 + EPS and (not times or times[-1] < t1 - EPS):
        times.append(t1)
    return times


def save_animation(frames: list, path: str, fps: int) -> None:
    """Save frames as .gif or .mp4."""
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)
