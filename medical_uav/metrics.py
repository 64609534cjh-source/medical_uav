"""Per-episode business metrics (§10.2)."""
from medical_uav.env.config import EPS
from medical_uav.env.simulator import Simulator

BUSY_KINDS = ("fly", "load", "unload", "dispose")


def episode_metrics(sim: Simulator) -> dict:
    orders = sim.orders
    n = len(orders)
    delivered = [o for o in orders if o.status == "delivered"]
    on_time = [o for o in delivered if o.end_time - o.release <= o.limit + EPS]
    late = [o for o in delivered if o.end_time - o.release > o.limit + EPS]
    duration = max(sim.t, EPS)
    busy = sum(s.t1 - s.t0 for s in sim.segments if s.kind in BUSY_KINDS)

    def rate(count):
        return count / n if n else 0.0

    return {
        "n_orders": n,
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
    }
