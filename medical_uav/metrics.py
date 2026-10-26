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


def episode_record(env, episode_load, decisions, action_counts, action_minutes,
                   learner_seed=None, invalid_actions=0, terminated=True, truncated=False):
    """Detailed shared record for training boundaries and full-day policy evaluation.

    A truncated record is labelled incomplete, so it cannot masquerade as a full day.
    Use the explicit reset load, since env.load can retain its constructor value.
    """
    env = env.unwrapped
    sim = env.sim
    metrics = episode_metrics(sim)
    orders = [o for o in sim.orders if o.status != "future"]
    type_counts = {}
    for k, order_type in enumerate(ORDER_TYPES):
        group = [o for o in orders if o.type_idx == k]
        delivered = [o for o in group if o.status == "delivered"]
        late = [o for o in delivered if o.end_time > o.deadline + EPS]
        type_counts[order_type.name] = {
            "released": len(group), "delivered": len(delivered),
            "on_time": len(delivered) - len(late),
            "expired": sum(o.status == "expired" for o in group),
            "late_deliveries": len(late),
            "lateness_sum": sum(o.end_time - o.deadline for o in late),
        }
    segment_minutes = {kind: sum(s.t1 - s.t0 for s in sim.segments if s.kind == kind)
                       for kind in ("fly", "load", "unload", "dispose", "charge", "wait")}
    metrics.update({
        "learner_seed": learner_seed, "demand_seed": int(env.demand_seed),
        "episode_load": episode_load, "lam": float(env.lam), "map_seed": env.cfg.map_seed,
        "released": len(orders), "expired": sum(o.status == "expired" for o in orders),
        "late_deliveries": sum(c["late_deliveries"] for c in type_counts.values()),
        "lateness_sum": sum(c["lateness_sum"] for c in type_counts.values()),
        "type_counts": type_counts, "reward": float(sim.total_reward), "decisions": decisions,
        "action_counts": dict(action_counts), "action_minutes": dict(action_minutes),
        "segment_minutes": segment_minutes, "wait_minutes": segment_minutes["wait"],
        "handling_minutes": sum(segment_minutes[k] for k in ("load", "unload", "dispose")),
        "invalid_actions": invalid_actions, "terminated": bool(terminated),
        "truncated": bool(truncated), "complete_day": bool(terminated and not truncated),
    })
    return metrics


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
