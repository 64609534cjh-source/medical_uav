"""A finished day as a record for the HTML replay player, and the player's state at any time.

During the day, decision_record() is called before each step; afterwards day_record() collects
the map, the orders with their times, the drone's activities and the decisions, and
write_html() puts the record into the player (replay_player.html), a single file that opens
offline in any browser. state_at() is what the player shows at time t; the player's script
computes the same (stateAt) and the tests check both against the simulator.
"""
import bisect
import json
from pathlib import Path

import numpy as np

from medical_uav.env.actions import decode
from medical_uav.env.config import CHARGERS, DAY_START_HOUR, EPS, M, NODE_NAMES, ORDER_TYPES
from medical_uav.metrics import episode_metrics

TEMPLATE = Path(__file__).with_name("replay_player.html")


def _name(node: int) -> str:
    return NODE_NAMES[node]


def action_text(env, a: int) -> str:
    """The action in words, read from the state before it is taken."""
    sim = env.sim
    kind, arg = decode(a)
    if kind == "pick":
        o = env.slots[arg]
        return f"Pick up {ORDER_TYPES[o.type_idx].name} at {_name(o.origin)} for {_name(o.dest)}"
    if kind == "deliver":
        boxes = [o for o in sim.onboard if o.dest == arg]
        names = " and ".join(sorted({ORDER_TYPES[o.type_idx].name for o in boxes}))
        return f"Deliver {len(boxes)} {names} box{'es' if len(boxes) > 1 else ''} to {_name(arg)}"
    if kind == "charge":
        station, target = arg
        return f"Charge at {'C1 (M)' if station == M else 'C2'} to {target:.0f}%"
    return "Return to the medical center" if kind == "return" else "Wait for the next order"


def short_text(env, a: int) -> str:
    """A shorter name of the action, for the list of the network's choices."""
    kind, arg = decode(a)
    if kind == "pick":
        o = env.slots[arg]
        return f"Pick {ORDER_TYPES[o.type_idx].name} {_name(o.origin)}→{_name(o.dest)}"
    if kind == "deliver":
        return f"Deliver at {_name(arg)}"
    if kind == "charge":
        return f"Charge {'C1' if arg[0] == M else 'C2'} to {arg[1]:.0f}%"
    return "Return to M" if kind == "return" else "Wait"


def unload_slack(deadlines, arrive: float, handling: float) -> float:
    """Smallest margin to a deadline when boxes are unloaded one by one, earliest deadline first
    (as the simulator does), each taking `handling` minutes after arriving at `arrive`."""
    return min(d - (arrive + handling * (k + 1)) for k, d in enumerate(sorted(deadlines)))


def decision_context(env, a: int) -> list[str]:
    """A few facts about the state the action was chosen in (what it serves, how urgent, the
    battery); they describe the situation, not the network's reasons."""
    sim, cfg = env.sim, env.cfg
    kind, arg = decode(a)
    facts = []
    if kind == "pick":
        o = env.slots[arg]
        facts.append(f"Due in {o.deadline - sim.t:.0f} min, expires in {o.expiry - sim.t:.0f} min")
        fly = sim.travel_time(sim.location, o.origin)
        facts.append("Already at the pickup" if fly < EPS else
                     f"{fly:.0f} min flight to {_name(o.origin)}")
    elif kind == "deliver":
        boxes = [o for o in sim.onboard if o.dest == arg and o.status == "onboard"]
        if boxes:
            arrive = sim.t + sim.travel_time(sim.location, arg)
            slack = unload_slack([o.deadline for o in boxes], arrive, cfg.handling_min)
            facts.append(f"Box closest to its deadline: {slack:.0f} min to spare" if slack >= 0
                         else f"Box closest to its deadline: {-slack:.0f} min late")
        facts.append(f"Unloads {len(boxes)} of the {len(sim.onboard)} boxes on board")
    elif kind == "charge":
        station, target = arg
        arrival = sim.battery - sim.energy(sim.location, station, sim.load_kg())
        facts.append(f"Battery {sim.battery:.0f}% now, {arrival:.0f}% on arrival")
        facts.append(f"{(target - arrival) / cfg.charge_rate:.0f} min of charging")
    n = len(sim.pending)
    facts.append(f"{n} order{'s' if n != 1 else ''} waiting, "
                 f"{len(sim.onboard)}/{cfg.max_boxes} boxes on board")
    return facts


def decision_record(env, a: int, probs: np.ndarray | None = None) -> dict:
    """One decision, recorded before env.step(a): its time, the action in words, the situation
    and, for a network, its three most likely actions with their probabilities."""
    top = []
    if probs is not None:
        top = [[short_text(env, int(k)), float(probs[k])] for k in np.argsort(probs)[::-1][:3]
               if probs[k] >= 0.005]
    return {"t": env.sim.t, "action": int(a), "text": action_text(env, a),
            "short": short_text(env, a), "facts": decision_context(env, a), "top": top}


def day_record(env, decisions: list[dict], policy: str) -> dict:
    """Everything the player needs about a finished day; `decisions` from decision_record()."""
    sim, cm, cfg = env.sim, env.city_map, env.cfg
    ends = [d["t"] for d in decisions[1:]] + [sim.t]
    decisions = [{**d, "end": end, "reward": r} for d, end, (_, _, r) in
                 zip(decisions, ends, env.decision_log)]
    flights = {(s.node_from, s.node_to) for s in sim.segments if s.kind == "fly"}
    m = episode_metrics(sim)
    return {
        "meta": {"policy": policy, "load": env.load, "seed": env.demand_seed,
                 "map_seed": cfg.map_seed, "end": sim.t, "release_window": cfg.release_window,
                 "peaks": [list(w) for w in cfg.peak_windows], "reserve": cfg.reserve,
                 "day_start_hour": DAY_START_HOUR, "max_boxes": cfg.max_boxes,
                 "max_kg": cfg.max_weight,
                 "summary": {"orders": m["n_orders"], "on_time": m["on_time"],
                             "delivered": m["delivered"],
                             "expired": sum(o.status == "expired" for o in sim.orders),
                             "km": m["km_flown"]}},
        "map": {"size": cfg.grid_size * cfg.cell_km,
                "nodes": [{"name": _name(i), "x": float(x), "y": float(y),
                           "role": "center" if i == M else
                           ("charger" if i in CHARGERS else "hospital")}
                          for i, (x, y) in enumerate(cm.node_xy)],
                "zones": [[c * cfg.cell_km, r * cfg.cell_km, w * cfg.cell_km, h * cfg.cell_km]
                          for r, c, h, w in cm.zones],
                "routes": {f"{a}-{b}": cm.paths[(a, b)].tolist() for a, b in sorted(flights)}},
        "types": [{"name": t.name, "limit": t.limit} for t in ORDER_TYPES],
        "orders": [{"id": o.id, "type": o.type_idx, "from": o.origin, "to": o.dest,
                    "kg": o.weight, "release": o.release, "limit": o.limit,
                    "deadline": o.deadline, "expiry": o.expiry, "pickup": o.pickup_time,
                    "end": o.end_time, "disposed": o.disposed_time, "status": o.status}
                   for o in sim.orders if o.status != "future"],
        "segments": [[s.kind, s.t0, s.t1, s.node_from, s.node_to, s.battery0, s.battery1,
                      s.order_id] for s in sim.segments],
        "decisions": decisions,
    }


def _along(path: list, f: float) -> tuple[float, float]:
    """The point at fraction f of a polyline's length."""
    p = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    goal = min(max(f, 0.0), 1.0) * lengths.sum()
    for i, d in enumerate(lengths):
        if goal <= d or i == len(lengths) - 1:
            g = min(1.0, goal / d) if d else 1.0
            return tuple(p[i] + g * (p[i + 1] - p[i]))
        goal -= d
    return tuple(p[-1])


def state_at(record: dict, t: float) -> dict:
    """What the player shows at time t: the drone's position and battery, the activity under
    way (index into the segments and the fraction done, or None), the ids of the boxes on board
    and of the waiting orders, the outcome counts so far, and the index of the current
    decision (-1 before the first)."""
    segs, nodes = record["segments"], record["map"]["nodes"]
    i = bisect.bisect_right([s[1] for s in segs], t + EPS) - 1  # last activity started by t
    xy, battery = (nodes[M]["x"], nodes[M]["y"]), (segs[0][5] if segs else 100.0)
    active, f = None, 1.0
    if i >= 0:
        kind, t0, t1, a, b, b0, b1 = segs[i][:7]
        f = min(1.0, max(0.0, (t - t0) / (t1 - t0))) if t1 > t0 else 1.0  # fraction done
        battery = b0 + f * (b1 - b0)
        xy = (_along(record["map"]["routes"][f"{a}-{b}"], f) if kind == "fly" and f < 1
              else (nodes[b]["x"], nodes[b]["y"]))
        active = i if f < 1 else None
    cargo, waiting, on_time, late, expired = [], [], 0, 0, 0
    for o in record["orders"]:
        if o["release"] > t + EPS:  # not yet known at time t
            continue
        picked = o["pickup"] is not None and o["pickup"] <= t + EPS
        closed = o["end"] is not None and o["end"] <= t + EPS
        if o["status"] == "delivered" and closed:
            if o["end"] - o["release"] <= o["limit"] + EPS:
                on_time += 1
            else:
                late += 1
            continue
        if o["status"] == "expired" and closed:
            expired += 1
        if picked:  # on board until delivered or, if expired, unloaded
            if not (o["disposed"] is not None and o["disposed"] <= t + EPS):
                cargo.append(o["id"])
        elif not closed:
            waiting.append(o["id"])
    decision = bisect.bisect_right([d["t"] for d in record["decisions"]], t + EPS) - 1
    return {"xy": xy, "battery": battery, "active": active, "f": f, "cargo": sorted(cargo),
            "waiting": sorted(waiting), "on_time": on_time, "late": late, "expired": expired,
            "decision": decision}


def opening_decision(record: dict) -> int:
    """The featured decision of the still and of the backlog video: the network's closest call
    between delivering and charging, else its closest call, else the first decision in the
    first demand peak."""
    decs = record["decisions"]
    calls = sorted((i for i, d in enumerate(decs) if len(d["top"]) > 1),
                   key=lambda i: decs[i]["top"][0][1] - decs[i]["top"][1][1])
    tradeoff = [i for i in calls if decs[i]["text"].startswith("Deliver")
                and decs[i]["top"][1][0].startswith("Charge")]
    if tradeoff or calls:
        return (tradeoff or calls)[0]
    peak = record["meta"]["peaks"][0][0]
    return next((i for i, d in enumerate(decs) if d["t"] >= peak), 0)


def write_html(record: dict, path) -> Path:
    """Write the player with the record inside; returns the path."""
    # "</" escaped so text in the record cannot close the player's <script> tag
    data = json.dumps(record, separators=(",", ":")).replace("</", "<\\/")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text().replace("/*DATA*/null", data))
    return path
