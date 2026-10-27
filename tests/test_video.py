import numpy as np
import pytest

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv


@pytest.fixture(scope="module")
def low_day():
    from medical_uav.env.replay import day_record, decision_record

    env = MedicalUAVEnv(load="low")
    env.reset(options={"demand_seed": 100_000})
    decisions, done = [], False
    while not done:
        a = slack_aware_policy(env)
        decisions.append(decision_record(env, a))
        _, _, done, _, _ = env.step(a)
    return day_record(env, decisions, "Slack-aware rule")


def test_frames_are_720p_rgb_at_any_time(low_day):
    from medical_uav.env.video import H, W, ReplayFrames

    renderer = ReplayFrames(low_day, "Low load")
    for t in (0.0, low_day["decisions"][5]["t"] + 0.5, low_day["meta"]["end"]):
        frame = renderer.draw(t)
        assert frame.shape == (H, W, 3) and frame.dtype == np.uint8


def test_smart_speed_flies_slowly_holds_decisions_and_ends_at_the_end(low_day):
    from medical_uav.env.replay import state_at
    from medical_uav.env.video import FLY_RATE, GROUND_RATE, smart_times

    fps = 12
    times = smart_times(low_day, fps)
    assert times[-1] == low_day["meta"]["end"] and all(b >= a for a, b in zip(times, times[1:]))
    first = low_day["decisions"][1]["t"]
    assert times.count(first) >= round(0.6 * fps)  # each decision held
    steps = [(a, b - a) for a, b in zip(times, times[1:]) if b > a]
    flying = [d for a, d in steps if (s := state_at(low_day, a))["active"] is not None
              and low_day["segments"][s["active"]][0] == "fly"]
    assert flying and max(flying) <= FLY_RATE / fps + 1e-9
    assert max(d for _, d in steps) <= GROUND_RATE / fps + 1e-9


def test_explained_times_hold_before_play_the_action_and_hold_after(low_day):
    from medical_uav.env.video import explained_times

    d = low_day["decisions"][3]
    times = explained_times(low_day, 3, 1, fps=10, before_s=1, action_s=2, after_s=1)
    assert len(times) == 40
    assert times[:10] == [(d["t"], "Deciding")] * 10 and times[-1][1] == "Done"
    assert d["t"] <= times[15][0] <= d["end"] and times[15][1] is None
