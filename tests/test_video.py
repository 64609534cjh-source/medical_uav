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


def test_explained_times_show_the_steps_of_a_decision_then_play_its_action(low_day):
    from medical_uav.env.video import explained_times

    d = low_day["decisions"][3]
    times = explained_times(low_day, 3, 1, fps=10)
    names = [p[0] if isinstance(p, tuple) else p for _, p in times]
    assert names[0] == "Leaving" and {"Deciding", "Picked", "Moving"} <= set(names)
    assert all(t == d["t"] for t, p in times if p is not None)
    run = [t for t, p in times if p is None]
    assert run[0] == d["t"] and run[-1] < d["end"] and run == sorted(run)


@pytest.mark.parametrize("note", [None, ["Charging decision", "PPO charges to 100%",
                                      "Slack delivers at H2", "Two orders waiting"]])
def test_comparison_frames_are_720p_rgb(low_day, note):
    from medical_uav.env.video import ComparisonFrames

    renderer = ComparisonFrames(low_day, low_day)
    frame = renderer.draw(65, note)
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8


def test_compare_times_follow_either_flight_and_include_window_endpoints():
    from medical_uav.env.video import FLY_RATE, GROUND_RATE, compare_times

    left = {"segments": [["wait", 0, 2], ["fly", 2, 4], ["wait", 4, 10]]}
    right = {"segments": [["wait", 0, 3], ["fly", 3, 6], ["wait", 6, 10]]}
    times = compare_times(left, right, fps=10, start=1, end=9)
    assert times[0] == 1 and times[-1] == 9
    for a, b in zip(times, times[1:]):
        rate = FLY_RATE if 2 <= a < 6 else GROUND_RATE
        assert 0 < b - a <= rate / 10 + 1e-9
    assert any(b - a == pytest.approx(GROUND_RATE / 10) for a, b in zip(times, times[1:]))
    assert compare_times(left, right, fps=10, start=3, end=3) == [3]


def test_locked_test_summary_uses_shipped_model_and_daily_rates(tmp_path, monkeypatch):
    from scripts import compare_video

    path = tmp_path / "map0.csv"
    path.write_text("method,load,seed,on_time_rate\n"
                    "m3_set_s11,high,1,0.8\nm3_set_s11,high,2,0.9\n"
                    "slack,high,1,0.6\nslack,high,2,0.7\n"
                    "m3_set_s11,medium,1,0.9\nm3_set_s11,medium,2,0.95\n"
                    "slack,medium,1,0.7\nslack,medium,2,0.9\n"
                    "m3_set_s11,low,1,1.0\nslack,low,1,1.0\n"
                    "m3_set_s11/ckpt_06000640,high,1,0.1\n"
                    "m3_set_s22,medium,1,0.1\nedf,high,1,0.1\n")
    expected = ["Locked test, 2 high-load days: PPO 85.0% vs Slack 65.0% on time",
                "Locked test, 2 medium-load days: PPO 92.5% vs Slack 80.0% on time"]
    assert compare_video.locked_test_summary(path) == expected
    monkeypatch.setattr(compare_video, "LOCKED_TEST", path)
    assert compare_video.locked_test_summary() == expected


def test_locked_test_summary_omits_results_when_file_is_absent(tmp_path, monkeypatch):
    from scripts import compare_video

    path = tmp_path / "absent.csv"
    assert compare_video.locked_test_summary(path) == []
    monkeypatch.setattr(compare_video, "LOCKED_TEST", path)
    assert compare_video.locked_test_summary() == []
