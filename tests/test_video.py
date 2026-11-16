import json
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


@pytest.mark.parametrize("caption", [None, ("Watch the backlog", ["One line"], False),
                                     ("PAUSED · 10:27", ["The rule charges", "PPO flies"], True)])
def test_duel_frames_are_1080p_rgb(low_day, caption):
    from medical_uav.env.video import DuelFrames

    renderer = DuelFrames(low_day, low_day)
    for t in (0.0, 65.0, low_day["meta"]["end"]):
        frame = renderer.draw(t, caption)
        assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8
    for step in range(len(renderer.GUIDE)):
        assert renderer.guide(65.0, step).shape == (1080, 1920, 3)


def test_duel_frames_show_the_steps_of_a_decision(low_day):
    from medical_uav.env.video import DuelFrames

    renderer = DuelFrames(low_day, low_day)
    t = low_day["decisions"][5]["t"]
    frames = [renderer.draw(t, None, (p, None)) for p in ("Deciding", "Picked", None)]
    assert all(f.shape == (1080, 1920, 3) for f in frames)
    assert (frames[0] != frames[1]).any() and (frames[1] != frames[2]).any()


def test_duel_series_follow_the_replay_state(low_day):
    from medical_uav.env.replay import state_at
    from medical_uav.env.video import DuelFrames

    renderer = DuelFrames(low_day, low_day)
    minutes, on_time, waiting = renderer.series[0]
    for k in (0, len(minutes) // 2, len(minutes) - 1):
        s = state_at(low_day, minutes[k])
        assert on_time[k] == s["on_time"] and waiting[k] == len(s["waiting"])


def test_day_times_follow_the_slower_side_and_cover_the_day(low_day):
    from medical_uav.env.video import DAY_GROUND, DAY_RATES, day_times

    end = low_day["meta"]["end"]
    times = day_times([low_day, low_day], fps=10, end=end)
    assert times[0] == 0 and times[-1] == end and times == sorted(times)
    steps = np.diff(times)
    assert steps.max() <= DAY_GROUND / 10 + 1e-9
    assert np.isclose(steps, DAY_RATES["fly"] / 10).any()


def test_slides_are_1080p_rgb(low_day):
    from medical_uav.env.video import DuelFrames

    renderer = DuelFrames(low_day, low_day)
    slides = [renderer.outline(["A", "B"]), renderer.outline(["A", "B"], 1, "note"),
              renderer.method("Net", [("A", "a"), ("B", "b")], ["line"]),
              renderer.bullets("Summary", ["one", "two"], "SUMMARY")]
    slides += [renderer.task(k, 65.0) for k in range(len(renderer.TASK))]
    assert all(f.shape == (1080, 1920, 3) for f in slides)


def test_replay_frames_scale_and_caption(low_day):
    from medical_uav.env.video import ReplayFrames

    renderer = ReplayFrames(low_day, "Low load", scale=1.5)
    renderer.focus = {"nodes": (1,), "orders": tuple(o["id"] for o in low_day["orders"][:2])}
    frame = renderer.draw(65.0, "Deciding", "", ("Title", "One line"))
    assert frame.shape == (1080, 1920, 3)


def test_final_rows_read_unseen_map_rates(tmp_path):
    from scripts import compare_video

    means = {k: [0.9, 0.95, 0.05] for k in ("slack", "insertion", "rollout", "scenario16")}
    blocks = {load: {"means": {**means, "final": [0.927, 0.96, 0.03],
                               "horizon": [0.932, 0.97, 0.02]}}
              for load in ("medium", "high")}
    blocks["high"]["means"]["final"] = [0.817, 0.87, 0.13]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"maps": blocks}))
    rows = compare_video.final_rows(path)
    assert [r[0] for r in rows][:2] == ["PPO (ours)", "Rolling horizon"]
    assert rows[0][2:4] == pytest.approx((92.7, 81.7))


def toy(orders, segments, end=100):
    return {"meta": {"end": end, "peaks": [[20, 60]]}, "orders": orders, "segments": segments,
            "map": {"nodes": [{"name": n} for n in ("M", "H1", "H2")]}}


def order(i, status, end, pickup=None, deadline=50, kind=0):
    return {"id": i, "type": kind, "from": 0, "to": 1, "release": 0, "limit": deadline,
            "deadline": deadline, "pickup": pickup, "end": end, "status": status}


def test_moments_name_real_events_from_both_days():
    from scripts import compare_video

    ppo = toy([order(1, "delivered", 40), order(2, "expired", 70)],
              [["charge", 10, 15, 0, 0, 60, 100, None], ["fly", 15, 100, 0, 1, 100, 50, None]])
    rule = toy([order(1, "expired", 80, pickup=30), order(2, "delivered", 45)],
               [["fly", 0, 30, 0, 1, 100, 20, None], ["charge", 30, 42, 1, 0, 16, 80, None]])
    loss = compare_video.ppo_failure(ppo, rule)
    assert loss[0] == 70 and "#2" in loss[1]
    charge = compare_video.rule_charge(ppo, rule, waiting=lambda r, t: 6)
    assert charge[0] == 30 and "16%" in charge[2][0] and "12-minute" in charge[1]


def test_record_day_keeps_the_network_top_choices_for_ppo_only():
    from scripts import compare_video

    ppo = compare_video.record_day("ppo", "low", 100000)
    slack = compare_video.record_day("slack", "low", 100000)
    assert all(d["top"] for d in ppo["decisions"])
    assert not any(d["top"] for d in slack["decisions"])
