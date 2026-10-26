import numpy as np

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import CHARGE_TABLE, CHARGE0, WAIT, charge_action, decode
from medical_uav.env.config import C2, FULL_BATTERY, M
from tests.conftest import make_order


def day_at(t: float, battery: float, load: str = "low") -> MedicalUAVEnv:
    """A day reset at the medical center, with the clock and battery set by hand."""
    env = MedicalUAVEnv(load=load)
    env.reset(options={"demand_seed": 100_000})
    env.sim.t, env.sim.battery = t, battery
    env._update_view()  # masks for the edited state
    return env


def test_without_full_charge_hides_exactly_the_full_charges():
    from medical_uav.baselines.energy import FULL_CHARGES, without_full_charge

    assert FULL_CHARGES == [charge_action(M, FULL_BATTERY), charge_action(C2, FULL_BATTERY)] \
        == [20, 23]
    env = day_at(30.0, 50.0)
    mask = env.action_masks()
    assert mask[FULL_CHARGES].all()
    seen = []
    policy = without_full_charge(lambda env, obs, m: seen.append(m.copy()) or WAIT)
    policy(env, None, mask)
    expected = mask.copy()
    expected[FULL_CHARGES] = False
    assert (seen[0] == expected).all() and seen[0][WAIT]
    assert mask[FULL_CHARGES].all()  # the caller's mask is unchanged


def test_slack100_changes_only_the_charge_target():
    from medical_uav.baselines.energy import slack100_policy

    charges = 0
    for k in range(40):
        env = MedicalUAVEnv(load="high")
        env.reset(options={"demand_seed": 100_000 + k % 10})
        for _ in range(10 + 2 * k):
            env.step(slack_aware_policy(env))
        a, b = slack_aware_policy(env), slack100_policy(env)
        kind, arg = decode(a)
        if kind == "charge" and env.action_masks()[charge_action(arg[0], FULL_BATTERY)]:
            charges += 1
            assert b == charge_action(arg[0], FULL_BATTERY)
        else:
            assert b == a
    assert charges > 0


def test_prepeak_top_up_boundaries():
    from medical_uav.baselines.energy import PrePeakTopUp

    full_at_m = charge_action(M, FULL_BATTERY)
    cases = [(30.0, 60.0, True),   # window opens at peak start 60 - 30
             (59.0, 60.0, False),  # 40 % more takes 8 min: would finish after the peak starts
             (60.0, 60.0, False),  # the peak has started
             (29.9, 60.0, False),  # before the window
             (40.0, 80.0, False)]  # not below 80 %
    for t, battery, override in cases:
        policy = PrePeakTopUp()
        a = policy(day_at(t, battery))
        assert (policy.used == {60.0}) == override, (t, battery)
        if override:
            assert a == full_at_m


def test_prepeak_top_up_needs_empty_cargo():
    from medical_uav.baselines.energy import PrePeakTopUp

    for status in ("onboard", "expired"):
        env = day_at(30.0, 60.0)
        box = make_order(0, 1, M, 2, 0.5, 0.0)
        box.status = status
        env.sim.onboard.append(box)
        env._update_view()
        policy = PrePeakTopUp()
        policy(env)
        assert policy.used == set(), status


def test_prepeak_top_up_once_per_peak_and_reset_each_day():
    from medical_uav.baselines.energy import PrePeakTopUp

    policy = PrePeakTopUp()
    env = day_at(30.0, 60.0)
    policy(env)
    env.sim.t, env.sim.battery = 40.0, 60.0  # still in the window, below 80 % again
    env._update_view()
    policy(env)
    assert policy.used == {60.0}
    env.sim.t, env.sim.battery = 275.0, 60.0  # second peak at 300
    env._update_view()
    assert policy(env) in (charge_action(M, FULL_BATTERY), charge_action(C2, FULL_BATTERY))
    assert policy.used == {60.0, 300.0}
    policy(day_at(30.0, 60.0))  # a new day
    assert policy.used == {60.0}


def test_full_charges_are_the_100_targets():
    from medical_uav.baselines.energy import FULL_CHARGES

    assert [CHARGE_TABLE[a - CHARGE0][1] for a in FULL_CHARGES] == [FULL_BATTERY] * 2
    assert np.all(np.diff(FULL_CHARGES) > 0)


def test_energy_contrasts_registered_verdict(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    keys = ["on_time_rate", "on_time_peak", "on_time_offpeak", "battery_at_peak",
            "peak_charge_minutes", "charge_minutes", "charge_full_share", "km_flown", "overrides"]
    base = dict(zip(keys, [0.8, 0.7, 0.9, 60.0, 30.0, 100.0, 0.1, 150.0, float("nan")]))
    changes = {"slack": {}, "slack100": {},
               "prepeak": {"on_time_peak": 0.75, "battery_at_peak": 70.0,
                           "peak_charge_minutes": 25.0, "overrides": 2},
               "ppo": {"on_time_peak": 0.9}, "ppo/no100": {"on_time_peak": 0.85}}
    lines = [",".join(["method", "load", "seed"] + keys)]
    for method, change in changes.items():
        for load in ("medium", "high"):
            for day in (1, 2, 3):
                row = {**base, **change}
                lines.append(",".join([method, load, str(day)] + [str(row[k]) for k in keys]))
    (tmp_path / "b.csv").write_text("\n".join(lines))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "energy_contrasts.py"),
                             str(tmp_path / "b.csv")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "peak on-time, prepeak - slack100:  +5.00 [ +5.00,  +5.00]" in result.stdout
    assert "VERDICT: anticipatory top-ups improve peak service" in result.stdout
    assert "ppo - no100:  +5.00" in result.stdout
