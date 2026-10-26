"""Waiting-time attribution, unchanged replay, and development separation."""
import json
from pathlib import Path

import pytest
import torch

from medical_uav.agents.networks import build_network
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import charge_action
from medical_uav.env.config import EnvConfig, M
from medical_uav.env.orders import Order
from medical_uav.evaluation.evaluate import evaluate
from medical_uav.evaluation.failures import FailureDiagnostics
from medical_uav.research.visibility import VisibilityAudit
from tests.conftest import make_sim


def test_order_released_and_expired_inside_charge_is_not_top_k_hidden(line_map):
    cfg = EnvConfig(release_window=10)
    order = Order(0, 0, M, 1, .8, 1., 1., 3.)
    env = MedicalUAVEnv(config=cfg)
    env.sim = make_sim(line_map, [order], cfg)
    env.sim.battery = 20
    env._update_view()
    observer = VisibilityAudit()
    action = charge_action(M, 100.)
    observer.before(env, action)
    env.step(action)
    observer.after(env, action)
    row = observer.finish(env)['visibility_audit']['orders'][0]
    assert env.sim.is_done()
    assert row['outcome'] == 'expired_unpicked'
    assert row['first_visible_delay_minutes'] is None
    assert row.get('hidden_decisions', 0) == 0
    assert row['pending_wait_minutes'] == pytest.approx(3.)
    assert row['unobserved_between_decisions_wait_minutes'] == pytest.approx(3.)
    env.close()


def test_additive_audit_does_not_change_complete_day_and_partitions_waiting():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = build_network('destination')
    other = build_network('structured')
    old, _ = evaluate(model, ('ppo',), (2,), 'train', ('high',), 3,
                      diagnostics=True, diagnostic_factory=FailureDiagnostics)
    new, _ = evaluate(model, ('ppo',), (2,), 'train', ('high',), 3,
                      diagnostics=True, diagnostic_factory=lambda: VisibilityAudit(other))
    record = new[0]
    detail = record['diagnostics'].pop('visibility_audit')
    assert record == old[0]
    assert len(detail['orders']) == record['released']
    assert detail['state_groups']['all']['states'] == record['decisions']
    for order in detail['orders']:
        summed = sum(order.get(k, 0.) for k in ['decision_held_visible_wait_minutes',
                                               'decision_held_hidden_wait_minutes',
                                               'unobserved_between_decisions_wait_minutes'])
        assert summed == pytest.approx(order['pending_wait_minutes'])
        assert order.get('legal_pick_not_chosen', 0) <= order.get('legal_pick_decisions', 0)


def test_registered_development_manifests_are_disjoint_and_no_new_test_exists():
    root = Path(__file__).parents[1]
    data = json.loads((root / 'configs/gen4_scenario_registry.json').read_text())
    tune = set(data['new_development']['dev_tune']['demand_seeds'])
    confirm = set(data['new_development']['dev_confirm']['demand_seeds'])
    assert len(tune) == len(confirm) == 100
    assert not tune & confirm
    forbidden = set(range(100000, 100140))
    for low, high in data['consumed_test_ranges_inclusive']:
        forbidden.update(range(low, high+1))
    assert min(tune | confirm) >= 100000 and not (tune | confirm) & forbidden
    assert data['new_final_test'] is None
    assert not set(sum(data['new_learner_seeds'].values(), [])) & set(range(6))
