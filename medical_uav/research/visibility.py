"""Read-only order visibility/waiting audit, never a policy input.

Visibility is recorded at decision times. Holding its label over a macro action
does not assert that the candidate set was continuously unchanged inside it.
"""
from collections import Counter

import torch

from medical_uav.env.actions import decode
from medical_uav.env.config import EPS, ORDER_TYPES
from medical_uav.evaluation.failures import FailureDiagnostics


class VisibilityAudit(FailureDiagnostics):
    def __init__(self, other_policy=None):
        super().__init__()
        self.other_policy = other_policy
        self.order_states = {}
        self.groups = {}

    def before(self, env, action):
        super().before(env, action)
        sim = env.sim
        visible = {o.id: i for i, o in enumerate(env.slots)}
        self.pending_before = list(sim.pending)
        self.visible_before = visible
        for order in self.pending_before:
            state = self.order_states.setdefault(order.id, Counter())
            shown = order.id in visible
            state['pending_decisions'] += 1
            state['visible_decisions' if shown else 'hidden_decisions'] += 1
            if shown:
                allowed = bool(env.action_masks()[visible[order.id]])
                state['legal_pick_decisions'] += allowed
                state['legal_pick_not_chosen'] += allowed and action != visible[order.id]
        conditions = ['all', 'overflow' if len(sim.pending) > 12 else 'no_overflow',
                      'urgent_live_cargo' if any(o.status == 'onboard' and o.deadline-sim.t <= 15
                                               for o in sim.onboard) else 'no_urgent_cargo',
                      'battery_below_40' if sim.battery < 40 else 'battery_at_least_40']
        other = None
        if self.other_policy is not None:
            with torch.inference_mode():
                other = int(self.other_policy.act(torch.from_numpy(env._obs.copy()),
                                                  torch.from_numpy(env.action_masks()), True)[0])
            if not env.action_masks()[other]:
                raise RuntimeError('counterfactual policy returned an illegal action')
        for condition in conditions:
            counter = self.groups.setdefault(condition, Counter())
            counter['states'] += 1
            counter['chosen_' + decode(action)[0]] += 1
            if other is not None:
                counter['other_' + decode(other)[0]] += 1
                counter['action_disagreements'] += action != other

    def after(self, env, action):
        super().after(env, action)
        for order in self.pending_before:
            end = order.pickup_time if order.pickup_time is not None else order.end_time
            until = min(env.sim.t, end if end is not None else env.sim.t)
            duration = max(0., until-self.before_t)
            key = ('decision_held_visible_wait_minutes' if order.id in self.visible_before
                   else 'decision_held_hidden_wait_minutes')
            self.order_states[order.id][key] += duration

    def finish(self, env):
        result = super().finish(env)
        orders = []
        for order in env.sim.orders:
            if order.status == 'future':
                continue
            state = dict(self.order_states.get(order.id, {}))
            first = self.first_seen.get(order.id)
            pending_end = order.pickup_time if order.pickup_time is not None else order.end_time
            if pending_end is None:
                raise RuntimeError('visibility audit requires complete days')
            pending_wait = max(0., pending_end-order.release)
            labelled = sum(state.get(k, 0.) for k in ('decision_held_visible_wait_minutes',
                                                     'decision_held_hidden_wait_minutes'))
            if labelled > pending_wait + 1e-7:
                raise RuntimeError('waiting-time partition exceeds pending duration')
            outcome = ('on_time' if order.status == 'delivered' and order.end_time <= order.deadline+EPS
                       else 'late' if order.status == 'delivered'
                       else 'expired_after_pick' if order.pickup_time is not None else 'expired_unpicked')
            orders.append({'order_id': order.id, 'type': ORDER_TYPES[order.type_idx].name,
                           'outcome': outcome, 'release': order.release, 'pickup_time': order.pickup_time,
                           'deadline': order.deadline, 'expiry': order.expiry, 'end_time': order.end_time,
                           'pending_wait_minutes': pending_wait,
                           'unobserved_between_decisions_wait_minutes': max(0., pending_wait-labelled),
                           'first_visible_delay_minutes': first['t']-order.release if first else None,
                           'first_visible_direct_slack_minutes': first['direct_slack'] if first else None,
                           **state})
        result['visibility_audit'] = {
            'orders': orders, 'state_groups': {k: dict(v) for k, v in self.groups.items()},
            'scope': 'observational; visibility at decision times only; decision-held waiting labels '
                     'are not continuous candidate occupancy; counterfactual queries share the exact '
                     'observation/mask but have no counterfactual outcome',
        }
        return result
