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
        "rejected": sum(1 for o in orders if o.status == "rejected"),
        "mean_lateness": (sum(o.end_time - o.deadline for o in late) / len(late)) if late else 0.0,
        "km_flown": sim.km_flown,
        "energy_used": sim.energy_used,
        "charge_minutes": sim.charging_time,
        "episode_minutes": sim.t,
        "charge_share": sim.charging_time / duration,
        "busy_share": busy / duration,
        "on_time_by_type": by_type,
    }


def share(count: int, total: int) -> str:
    """Return 'count/total (percent%)', the way rates are printed."""
    return f"{count}/{total} ({100 * count / total:.1f}%)" if total else f"{count}/0"


def summary_lines(m: dict) -> list[str]:
    """Return the main metrics of a day as short labelled lines for printing."""
    n = m["n_orders"]
    expired = round(m["expiry_rate"] * n)
    by_type = ", ".join(f"{name} {'n/a' if math.isnan(rate) else f'{100 * rate:.0f}%'}"
                        for name, rate in m["on_time_by_type"].items())
    lines = [
        f"Orders   {n} released: delivered {share(m['delivered'], n)}, "
        f"expired {share(expired, n)}, rejected {share(m['rejected'], n)}",
        f"On time  {share(m['on_time'], n)}; late {m['delivered'] - m['on_time']} "
        f"(mean {m['mean_lateness']:.1f} min late)",
        f"By type  on time: {by_type}",
        f"Effort   {m['km_flown']:.1f} km flown, charging {m['charge_minutes']:.1f} min "
        f"({100 * m['charge_share']:.1f}% of the day)",
    ]
    if "return" in m:
        lines.append(f"Reward   {m['return']:.2f} in total")
    return lines
