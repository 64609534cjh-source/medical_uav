"""Insertion heuristic: keep a route of pickups and deliveries, insert each visible order where
the route serves the most orders on time, and fly to the route's first stop."""
from medical_uav.baselines.heuristics import DEFAULT_THETA, charging_rule, slack_aware_policy
from medical_uav.env.actions import DELIVER0, PICK0, WAIT
from medical_uav.env.config import EPS, FACILITIES

# A stop is ("pick", order) or ("deliver", node); a deliver stop unloads every box for node.


def route_score(sim, route) -> tuple[int, int, float] | None:
    """Simulate the route from the current state and return (on-time deliveries, deliveries,
    minus the sum of delivery times), or None if a stop breaks the cargo limits, picks an order
    after it expires or delivers to a node with no box for it. Battery is not planned."""
    cfg, h = sim.cfg, sim.cfg.handling_min
    t, loc = sim.t, sim.location
    cargo = list(sim.onboard)  # expired boxes are dropped at the first facility
    on_time = served = 0
    total = 0.0
    for kind, arg in route:
        node = arg.origin if kind == "pick" else arg
        t += sim.travel_time(loc, node)
        loc = node
        if node in FACILITIES:  # expired boxes are unloaded on arrival
            expired = [o for o in cargo if o.expiry <= t + EPS]
            t += h * len(expired)
            cargo = [o for o in cargo if o not in expired]
        if kind == "pick":
            if len(cargo) + 1 > cfg.max_boxes:
                return None
            if sum(o.weight for o in cargo) + arg.weight > cfg.max_weight + EPS:
                return None
            if t + h >= arg.expiry - EPS:
                return None
            t += h
            cargo.append(arg)
            continue
        boxes = sorted((o for o in cargo if o.dest == node), key=lambda o: (o.deadline, o.id))
        if not boxes:
            return None
        for o in boxes:
            t += h
            if t < o.expiry - EPS:  # a box expiring during its unload is lost
                served += 1
                on_time += t <= o.deadline + EPS
                total += t
        cargo = [o for o in cargo if o.dest != node]
    return on_time, served, -total


def _best(sim, candidates):
    """Return the best-scoring valid route among the candidates, with its score."""
    best, best_score = None, None
    for route in candidates:
        score = route_score(sim, route)
        if score is not None and (best_score is None or score > best_score):
            best, best_score = route, score
    return best, best_score


def _insertions(route, order):
    """Yield every route with the order's pickup inserted and its delivery after it, either at
    an existing stop for its destination or at a new stop."""
    for i in range(len(route) + 1):
        with_pick = route[:i] + [("pick", order)] + route[i:]
        if ("deliver", order.dest) in with_pick[i + 1:]:
            yield with_pick
        for j in range(i + 1, len(with_pick) + 1):
            yield with_pick[:j] + [("deliver", order.dest)] + with_pick[j:]


def plan_route(env, late: str = "end") -> list:
    """Build the route: deliveries for the cargo first, then each visible order by deadline at
    its best position, kept only if it adds an on-time delivery. Orders that add only a late
    delivery are inserted only at the end of the route (late="end", chosen on the screen days),
    anywhere ("insert") or not at all ("skip")."""
    sim = env.sim
    route = []
    cargo = [o for o in sim.onboard if o.status == "onboard"]
    for node in sorted({o.dest for o in cargo},
                       key=lambda n: min(o.deadline for o in cargo if o.dest == n)):
        best, _ = _best(sim, (route[:i] + [("deliver", node)] + route[i:]
                              for i in range(len(route) + 1)))
        if best is not None:
            route = best
    score = route_score(sim, route)
    deferred = []
    for order in sorted(env.slots, key=lambda o: (o.deadline, o.id)):
        best, best_score = _best(sim, _insertions(route, order))
        if best is not None and best_score[0] > score[0]:
            route, score = best, best_score
        elif (late == "insert" and best is not None and best_score[0] == score[0]
              and best_score[1] > score[1]):
            route, score = best, best_score
        else:
            deferred.append(order)
    if late == "end":
        for order in deferred:
            tail = route + [("pick", order), ("deliver", order.dest)]
            tail_score = route_score(sim, tail)
            if tail_score is not None and tail_score[1] > score[1]:
                route, score = tail, tail_score
    return route


def insertion_policy(env, theta: float = DEFAULT_THETA, late: str = "end") -> int:
    """Charge if due, else take the first stop of the insertion route, else WAIT. If that stop
    is not allowed now (battery), fall back to Slack-aware."""
    a = charging_rule(env, theta)
    if a is not None:
        return a
    route = plan_route(env, late)
    if not route:
        return WAIT
    kind, arg = route[0]
    a = PICK0 + env.slots.index(arg) if kind == "pick" else DELIVER0 + arg
    return a if env.action_masks()[a] else slack_aware_policy(env, theta)
