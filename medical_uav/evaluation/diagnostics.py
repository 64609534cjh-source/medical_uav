"""Read-only episode diagnostics; neither model selection nor simulator semantics."""

from collections import Counter

import numpy as np

from medical_uav.env.actions import decode
from medical_uav.env.config import EPS, ORDER_TYPES


def histogram(values, edges):
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    if len(values) and int(counts.sum()) != len(values):
        raise ValueError("resource observation outside the declared histogram bounds")
    return {"edges": list(edges), "counts": counts.tolist(), "n": len(values),
            "sum": float(values.sum()),
            "quantiles_per_episode": np.quantile(values, [.05, .25, .5, .75, .95]).tolist() if len(values) else None}


class EpisodeDiagnostics:
    def __init__(self):
        self.battery, self.payload, self.boxes = [], [], []
        self.delivery_sizes = []
        self.prior_events = 0

    def before(self, env, action):
        sim = env.sim
        self.battery.append(sim.battery)
        self.payload.append(sim.load_kg())
        self.boxes.append(len(sim.onboard))
        self.prior_events = len(sim.events)

    def after(self, env, action):
        if decode(action)[0] == "deliver":
            self.delivery_sizes.append(sum(e[1] == "deliver" for e in env.sim.events[self.prior_events:]))

    def finish(self, env):
        sim = env.sim
        orders = [o for o in sim.orders if o.status != "future"]
        outcome = {}
        for k, kind in enumerate(ORDER_TYPES):
            group = [o for o in orders if o.type_idx == k]
            outcome[kind.name] = {
                "released": len(group), "picked": sum(o.pickup_time is not None for o in group),
                "on_time": sum(o.status == "delivered" and o.end_time <= o.deadline + EPS for o in group),
                "late": sum(o.status == "delivered" and o.end_time > o.deadline + EPS for o in group),
                "expired_unpicked": sum(o.status == "expired" and o.pickup_time is None for o in group),
                "expired_after_pick": sum(o.status == "expired" and o.pickup_time is not None for o in group),
            }
        # Reconstruct physical flight payload from complete immutable order histories.
        # Expired cargo remains aboard until disposal, not merely until expiry.
        flight_battery, flight_kg, flight_boxes = [], [], []
        for segment in sim.segments:
            if segment.kind != "fly":
                continue
            cargo = [o for o in orders if o.pickup_time is not None and o.pickup_time <= segment.t0 + EPS
                     and (o.end_time if o.status == "delivered" else o.disposed_time if o.disposed_time is not None else float("inf")) > segment.t0 + EPS]
            kg = sum(o.weight for o in cargo)
            predicted = sim.energy(segment.node_from, segment.node_to, kg)
            if not np.isclose(predicted, segment.battery0 - segment.battery1, atol=1e-6, rtol=1e-7):
                raise RuntimeError("reconstructed flight payload disagrees with recorded energy")
            flight_battery.append(segment.battery0); flight_kg.append(kg); flight_boxes.append(len(cargo))
        effective = [n for n in self.delivery_sizes if n > 0]
        totals = {key: sum(v[key] for v in outcome.values()) for key in next(iter(outcome.values()))}
        assert totals["released"] == totals["on_time"] + totals["late"] + totals["expired_unpicked"] + totals["expired_after_pick"]
        assert totals["picked"] == totals["on_time"] + totals["late"] + totals["expired_after_pick"]
        return {"outcomes": totals, "outcomes_by_type": outcome,
                "delivery": {"actions": len(self.delivery_sizes), "effective_actions": len(effective),
                             "empty_actions": len(self.delivery_sizes) - len(effective),
                             "delivered_boxes": sum(effective), "batch_actions": sum(n >= 2 for n in effective)},
                "resources": {
                    "decision_battery": histogram(self.battery, np.linspace(0, 100, 21).tolist()),
                    "decision_payload_kg": histogram(self.payload, np.linspace(0, env.cfg.max_weight + 1e-6, 11).tolist()),
                    "departure_battery": histogram(flight_battery, np.linspace(0, 100, 21).tolist()),
                    "departure_payload_kg": histogram(flight_kg, np.linspace(0, env.cfg.max_weight + 1e-6, 11).tolist()),
                    "departure_boxes": histogram(flight_boxes, [-.5, .5, 1.5, 2.5, 3.5]),
                }}
