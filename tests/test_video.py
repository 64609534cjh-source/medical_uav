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


@pytest.mark.parametrize("paused", [False, True])
def test_duel_frames_are_1080p_rgb(low_day, paused):
    from medical_uav.env.video import DuelFrames

    renderer = DuelFrames(low_day, low_day)
    for t in (0.0, 65.0, low_day["meta"]["end"]):
        frame = renderer.draw(t, paused)
        assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8
    for step in range(len(renderer.GUIDE)):
        assert renderer.guide(65.0, step, "a note\nsecond line").shape == (1080, 1920, 3)


def test_captions_are_drawn_in_the_frame_and_wrap_english_and_chinese():
    from matplotlib import font_manager
    from medical_uav.env.video import Captions

    fonts = [font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=w))
             for w in ("bold", "normal")]
    captions = Captions(*fonts)
    frame = np.full((1080, 1920, 3), 240, np.uint8)
    for place in Captions.PLACES:
        out = captions.draw(frame, "Title", "A line long enough to wrap " * 8, False, place)
        assert out.shape == frame.shape and (out != frame).any()
    assert captions.draw(frame, "") is frame  # no caption
    font = captions.font(fonts[1], 28)
    assert len(Captions.wrap("word " * 60, font, 400)) > 2
    assert len(Captions.wrap("中文没有空格也要折行" * 10, font, 400)) > 2


def test_duel_frames_light_up_a_new_decision(low_day):
    from medical_uav.env.video import DuelFrames

    renderer = DuelFrames(low_day, low_day)
    t = low_day["decisions"][5]["t"]
    frames = [renderer.draw(t, None, (p, None)) for p in ("Deciding", None)]
    assert all(f.shape == (1080, 1920, 3) for f in frames) and (frames[0] != frames[1]).any()


def test_duel_series_follow_the_replay_state(low_day):
    from medical_uav.env.replay import state_at
    from medical_uav.env.video import DuelFrames, flown

    renderer = DuelFrames(low_day, low_day)
    minutes, (on_time, waiting, km) = renderer.minutes, renderer.series[0]
    for k in (0, len(minutes) // 2, len(minutes) - 1):
        state = state_at(low_day, min(minutes[k], low_day["meta"]["end"]))
        assert waiting[k] == len(state["waiting"]) and on_time[k] == state["on_time"]
    assert km[0] == 0 and np.all(np.diff(km) >= 0) and np.all(np.diff(on_time) >= 0)
    end = low_day["meta"]["end"]
    assert flown(low_day, end) == pytest.approx(low_day["meta"]["summary"]["km"])
    assert km[-1] == pytest.approx(flown(low_day, end, True)) and 0 < km[-1] < flown(low_day, end)


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
    ids = tuple((o["id"], str(j + 1)) for j, o in enumerate(low_day["orders"][:2]))
    slides = [renderer.bullets("Summary", ["one", "two"], "SUMMARY"),
              renderer.versus([("A", "a line", ""), ("B", "b line", "badge")], "foot"),
              renderer.day_numbers(), renderer.ways([("A", "a"), ("B", "b")], 1),
              renderer.close(65.0, 30.0, ((1,), (1,)), ids, "SLOW MOTION")]
    assert all(f.shape == (1080, 1920, 3) for f in slides)


def test_replay_frames_scale_and_spotlight(low_day):
    from medical_uav.env.video import ReplayFrames

    renderer = ReplayFrames(low_day, "Low load", scale=1.5)
    renderer.focus = {"nodes": (1,), "orders": tuple(o["id"] for o in low_day["orders"][:2])}
    frame = renderer.draw(65.0, "Deciding", "", ([(10, 10, 100, 100)], 0.5))
    assert frame.shape == (1080, 1920, 3)


def test_final_rows_read_unseen_map_rates(tmp_path):
    from scripts import compare_video

    means = {k: [0.9, 0.95, 0.05] for k in ("slack", "insertion", "rollout", "scenario16",
                                            "nearest", "edf", "fcfs", "random")}
    blocks = {load: {"means": {**means, "final": [0.927, 0.96, 0.03],
                               "horizon": [0.932, 0.97, 0.02]}}
              for load in ("medium", "high")}
    blocks["high"]["means"]["final"] = [0.817, 0.87, 0.13]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"maps": blocks}))
    rows = compare_video.final_rows(path)
    assert [r[0] for r in rows][:2] == ["PPO (ours)", "Rolling horizon"]
    assert rows[0][2:4] == pytest.approx((92.7, 81.7))


def test_record_day_keeps_the_network_top_choices_for_ppo_only():
    from scripts import compare_video

    ppo = compare_video.record_day("ppo", "low", 100000)
    slack = compare_video.record_day("slack", "low", 100000)
    assert all(d["top"] for d in ppo["decisions"])
    assert not any(d["top"] for d in slack["decisions"])


def test_stretches_follow_each_other_and_their_boxes_are_picked_inside():
    from scripts import compare_video

    ppo, rule = (compare_video.record_day(n, "high", 100007) for n in ("ppo", "slack"))
    first, second = compare_video.stretches(ppo, rule)
    assert first["end"] == second["start"] and second["shown"] < second["start"]
    assert len(first["ids"]) == 3 and len(second["ids"]) == 3
    orders = {o["id"]: o for o in ppo["orders"]}
    for w in (first, second):
        assert all(w["start"] <= orders[k]["pickup"] < w["end"] for k, _ in w["ids"])
    for w in (first, second):
        times = [t for t, _ in w["moments"]]
        assert times == sorted(times) and all(w["start"] <= t <= w["end"] for t in times)

def test_morph_ends_on_each_frame():
    from medical_uav.env.video import morph

    a, b = np.zeros((90, 160, 3), np.uint8), np.full((90, 160, 3), 200, np.uint8)
    boxes = ([(0, 0, 40, 40)], [(60, 20, 80, 60)])
    assert (morph(a, b, *boxes, 0.0) == a).all() and (morph(a, b, *boxes, 1.0) == b).all()


def test_zoom_moves_smoothly_from_the_whole_frame_into_the_box():
    from medical_uav.env.video import zoom

    frame = np.zeros((90, 160, 3), np.uint8)
    frame[:45, :80] = 255  # the top-left quarter is white
    assert np.abs(zoom(frame, (0, 0, 80, 45), 0.0).astype(int) - frame).mean() < 2
    inside = zoom(frame, (0, 0, 80, 45), 1.0)
    assert inside.shape == frame.shape and inside[5:-5, 5:-5].min() > 240


def test_band_has_the_frame_width_and_slides_text_in():
    from matplotlib import font_manager
    from medical_uav.env.video import Band

    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    band = Band(bold)
    start, done = band.draw("Title", appear=0.0, flash=1.0), band.draw("Title")
    assert start.shape == done.shape == (Band.H, 1920, 3) and (start != done).any()
