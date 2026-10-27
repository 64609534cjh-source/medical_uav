import numpy as np

from medical_uav.baselines.horizon import RollingHorizon, foresight_root
from medical_uav.baselines.rollout import rollout_policy
from tests.test_scenario import busy_state, observable_view


def test_one_step_visible_optimizer_is_the_visible_order_rollout():
    for k in range(5):
        env = busy_state(seed=100_000 + k, steps=25 + 5 * k)
        assert RollingHorizon(depth=1)(env) == rollout_policy(env)


def test_visible_optimizer_reads_only_what_the_agent_observes():
    for k in range(3):
        env = busy_state(seed=100_010 + k, steps=30 + 5 * k)
        assert RollingHorizon().scores(observable_view(env)) == RollingHorizon().scores(env)


def test_foresight_adds_hidden_and_next_hour_orders():
    env = busy_state(steps=20)
    sim, root = env.sim, foresight_root(env).sim
    future = [o.id for o in sim.orders[sim._next_release:] if o.release <= sim.t + 60]
    assert future and [o.id for o in root.orders[root._next_release:]] == future
    assert sorted(o.id for o in root.pending) == sorted(o.id for o in sim.pending)
    assert [o.id for o in foresight_root(env).slots] == [o.id for o in env.slots]


def test_a_day_with_the_optimizer_takes_only_allowed_actions():
    from medical_uav.evaluation import heuristic, run_day

    row = run_day(heuristic(RollingHorizon(foresight=True)), "low", 100_000)
    assert row["decisions"] > 0 and 0 <= row["on_time_rate"] <= 1
