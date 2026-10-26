"""Operational metrics for one simulated day."""
import math

from medical_uav.env.config import EPS, ORDER_TYPES
from medical_uav.env.simulator import Simulator

BUSY_KINDS = ("fly", "load", "unload", "dispose")  # segment kinds counted as busy time


def episode_metrics(sim: Simulator) -> dict:
    """Return metrics over the orders released so far (all orders once the day is over).

    Mean lateness averages over late deliveries only.
    """
    orders = [o for o in sim.orders if o.status != "future"]
    n = len(orders)
    delivered = [o for o in orders if o.status == "delivered"]
    on_time = [o for o in delivered if o.end_time - o.release <= o.limit + EPS]
    late = [o for o in delivered if o.end_time - o.release > o.limit + EPS]
    duration = max(sim.t, EPS)
    busy = sum(s.t1 - s.t0 for s in sim.segments if s.kind in BUSY_KINDS)

    def rate(count):
        return count / n if n else 0.0

    by_type = {}  # on-time rate per order type; NaN for a type with no orders
    for k, order_type in enumerate(ORDER_TYPES):
        n_type = sum(1 for o in orders if o.type_idx == k)
        n_ok = sum(1 for o in on_time if o.type_idx == k)
        by_type[order_type.name] = n_ok / n_type if n_type else math.nan

    return {
        "n_orders": n,
        "delivered": len(delivered),
        "on_time": len(on_time),
        "on_time_rate": rate(len(on_time)),
        "served_rate": rate(len(delivered)),
        "expiry_rate": rate(sum(1 for o in orders if o.status == "expired")),
        "mean_lateness": (sum(o.end_time - o.deadline for o in late) / len(late)) if late else 0.0,
        "km_flown": sim.km_flown,
        "energy_used": sim.energy_used,
        "charge_minutes": sim.charging_time,
        "episode_minutes": sim.t,
        "charge_share": sim.charging_time / duration,
        "busy_share": busy / duration,
        "on_time_by_type": by_type,
    }


def battery_at(sim: Simulator, t: float) -> float:
    """Battery % at time t, from the recorded activities (unchanged between them)."""
    battery = sim.segments[0].battery0 if sim.segments else sim.battery
    for s in sim.segments:
        if s.t0 > t:
            break
        if t < s.t1:
            return s.battery0 + (s.battery1 - s.battery0) * (t - s.t0) / (s.t1 - s.t0)
        battery = s.battery1
    return battery


def behaviour_metrics(sim: Simulator, decision_log: list) -> dict:
    """How a day was served: boxes delivered per delivery stop (a box that expires while being
    unloaded does not count), charges, the battery when they started, the share of charges to
    100 %, the share of charging minutes inside the peaks, the battery when each peak began,
    the shares of WAIT and RETURN decisions, and the on-time rate of orders released inside and
    outside the peaks."""
    stops = 0
    previous = None
    for seg in sim.segments:
        if seg.kind == "unload":
            if previous is None or previous.kind != "unload" or previous.node_to != seg.node_to:
                stops += 1  # a new stop starts unless the drone keeps unloading at the same node
        if seg.kind != "dispose":
            previous = seg
    delivered = sum(1 for _, kind, _ in sim.events if kind == "deliver")
    charges = [seg for seg in sim.segments if seg.kind == "charge"]
    labels = [entry[1] for entry in decision_log]
    charge_labels = [label for label in labels if label.startswith("CHARGE")]
    windows = sim.cfg.peak_windows
    charge_minutes = sum(c.t1 - c.t0 for c in charges)
    peak_minutes = sum(max(0.0, min(c.t1, b) - max(c.t0, a)) for c in charges for a, b in windows)
    peak_batteries = [battery_at(sim, a) for a, _ in windows if a <= sim.t]

    def on_time_rate(in_peak: bool) -> float:
        released = [o for o in sim.orders if o.status != "future"
                    and any(a <= o.release < b for a, b in sim.cfg.peak_windows) == in_peak]
        on_time = [o for o in released
                   if o.status == "delivered" and o.end_time - o.release <= o.limit + EPS]
        return len(on_time) / len(released) if released else math.nan

    return {
        "boxes_per_stop": delivered / stops if stops else math.nan,
        "charges": len(charges),
        "battery_at_charge": (sum(c.battery0 for c in charges) / len(charges)
                              if charges else math.nan),
        "charge_full_share": (sum(label.endswith(",100)") for label in charge_labels)
                              / len(charge_labels) if charge_labels else math.nan),
        "peak_charge_share": peak_minutes / charge_minutes if charge_minutes else math.nan,
        "battery_at_peak": (sum(peak_batteries) / len(peak_batteries)
                            if peak_batteries else math.nan),
        "wait_share": labels.count("WAIT") / len(labels) if labels else math.nan,
        "return_share": labels.count("RETURN") / len(labels) if labels else math.nan,
        "on_time_peak": on_time_rate(True),
        "on_time_offpeak": on_time_rate(False),
    }


def share(count: int, total: int) -> str:
    """Return 'count/total (percent%)', the way rates are printed."""
    return f"{count}/{total} ({100 * count / total:.1f}%)" if total else f"{count}/0"


def summary_lines(m: dict) -> list[str]:
    """Return the main metrics of a day as short labelled lines for printing."""
    n = m["n_orders"]
    expired = round(m["expiry_rate"] * n)
    rates = []
    for name, rate in m["on_time_by_type"].items():
        rates.append(f"{name} {'n/a' if math.isnan(rate) else f'{100 * rate:.0f}%'}")
    by_type = ", ".join(rates)
    lines = [
        f"Orders   {n} released: delivered {share(m['delivered'], n)}, "
        f"expired {share(expired, n)}",
        f"On time  {share(m['on_time'], n)}; late {m['delivered'] - m['on_time']} "
        f"(mean {m['mean_lateness']:.1f} min late)",
        f"By type  on time: {by_type}",
        f"Effort   {m['km_flown']:.1f} km flown, charging {m['charge_minutes']:.1f} min "
        f"({100 * m['charge_share']:.1f}% of the day)",
    ]
    if "return" in m:
        lines.append(f"Reward   {m['return']:.2f} in total")
    return lines
