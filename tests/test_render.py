import numpy as np
import pytest

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import M
from medical_uav.env.render import Renderer, drone_state_at
from tests.conftest import make_order, make_sim


def test_position_interpolates_along_flight(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])  # t = 2 at M
    sim.do_deliver(2)  # fly 2 -> 5 (3 km)
    s = drone_state_at(sim, 3.5)
    assert s["xy"] == pytest.approx(np.array([1.5, 0.0]))
    assert s["activity"] == "fly" and len(s["cargo"]) == 1
    assert s["battery"] == pytest.approx(100 - 1.5 * 3.5 * 1.125)


def test_cargo_empty_after_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.do_deliver(2)
    assert drone_state_at(sim, sim.t)["cargo"] == []


def test_pending_visible_only_after_release(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 5.0)])
    sim.advance_to(10.0)
    assert drone_state_at(sim, 4.0)["pending"] == []
    assert [o.id for o in drone_state_at(sim, 6.0)["pending"]] == [0]


def test_draw_returns_rgb_frame(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    frame = Renderer(line_map, sim.cfg).draw(sim, 0.0)
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3


def test_env_render_rgb_array():
    env = MedicalUAVEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(25)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_frame_times_every_minute_then_the_end():
    from medical_uav.env.render import frame_times

    assert frame_times(3.0) == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert frame_times(2.5) == pytest.approx([0.0, 1.0, 2.0, 2.5])
    assert frame_times(0.0) == [0.0]


def test_save_animation_streams_generator(tmp_path):
    import imageio.v2 as imageio

    from medical_uav.env.render import save_animation

    # distinct frames: GIF encoders merge identical consecutive frames
    frames = (np.full((32, 48, 3), 40 * i, dtype=np.uint8) for i in range(5))
    assert save_animation(frames, str(tmp_path / "a.gif"), fps=5) == 5
    assert len(imageio.mimread(tmp_path / "a.gif")) == 5


def test_draw_accepts_title_and_decision(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    renderer = Renderer(line_map, sim.cfg, title="Slack-aware | medium load | demand seed 0")
    frame = renderer.draw(sim, 0.0, decision="Last decision 08:00: WAIT")
    assert frame.shape == (528, 960, 3)  # height divisible by 16 for video codecs


def test_activity_text_follows_the_segment_at_t(line_map):
    from medical_uav.env.render import activity_text

    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])  # load at M during [0, 2]
    sim.do_deliver(2)  # fly M -> 2, then unload
    by_id = {o.id: o for o in sim.orders}

    def text(t):
        return activity_text(drone_state_at(sim, t), by_id, sim.location)

    assert text(1.0).startswith("Loading #0")
    assert text(3.5).startswith("Flying")
    assert text(sim.t - 1.0).startswith("Unloading #0")


def test_outcome_counts_use_the_history_not_the_final_state(line_map):
    from medical_uav.env.render import outcome_counts

    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.do_deliver(2)
    before, after = outcome_counts(sim, 3.0), outcome_counts(sim, sim.t)
    assert (before["delivered"], before["open"]) == (0, 1)
    assert (after["delivered"], after["on_time"], after["open"]) == (1, 1, 0)


def test_clock_starts_at_eight():
    from medical_uav.env.render import clock

    assert clock(0.0) == "08:00" and clock(222.4) == "11:42"


@pytest.mark.parametrize("ext", ["gif", "mp4"])
def test_animation_memory_does_not_grow_with_frames(tmp_path, ext):
    import tracemalloc

    from medical_uav.env.render import save_animation

    frame_mb = 208 * 304 * 3 / 1e6
    rng = np.random.default_rng(0)  # noisy frames: encoders cannot shrink them to almost nothing
    frames = (rng.integers(0, 256, (208, 304, 3), dtype=np.uint8) for _ in range(120))
    tracemalloc.start()
    save_animation(frames, str(tmp_path / f"m.{ext}"), fps=10)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / 1e6 < 8 * frame_mb  # a buffering writer would hold all 120 frames


def test_save_animation_mp4(tmp_path):
    import imageio.v2 as imageio

    from medical_uav.env.render import save_animation

    frames = (np.full((32, 48, 3), 40 * i, dtype=np.uint8) for i in range(4))
    assert save_animation(frames, str(tmp_path / "c.mp4"), fps=4) == 4
    assert imageio.get_reader(tmp_path / "c.mp4").count_frames() == 4


def test_save_animation_rejects_other_extensions(tmp_path):
    from medical_uav.env.render import save_animation

    with pytest.raises(ValueError):
        save_animation([np.zeros((8, 8, 3), dtype=np.uint8)], str(tmp_path / "d.png"), fps=4)


def test_every_order_type_has_a_colour():
    from medical_uav.env.config import ORDER_TYPES
    from medical_uav.env.render import TYPE_COLORS

    assert set(TYPE_COLORS) == {t.name for t in ORDER_TYPES}


LOG = [(10.0, "PICK medicine M->H2", 0.0), (15.0, "DELIVER(H2)", 1.0), (20.0, "WAIT", 0.0)]


def test_explanation_holds_each_decision():
    from medical_uav.env.render import explanation_times

    frames = explanation_times(LOG, 0, 3, hold_s=2.0, travel_s=3.0, fps=12, t_end=25.0)
    assert len(frames) >= 3 * (2 * 2 + 3) * 12
    for n, (t0, label, _) in enumerate(LOG, start=1):
        held = [t == t0 and label in caption and f"Decision {n}" in caption for t, caption in frames]
        longest, run = 0, 0
        for h in held:
            run = run + 1 if h else 0
            longest = max(longest, run)
        assert longest >= 24


def test_explanation_near_end_of_day():
    from medical_uav.env.render import explanation_times

    frames = explanation_times(LOG, 2, 8, hold_s=1.0, travel_s=1.0, fps=4, t_end=25.0)
    assert frames and all("Decision 3" in caption for _, caption in frames)
    assert min(t for t, _ in frames) == 20.0 and max(t for t, _ in frames) == 25.0


def test_outcome_text_reports_deliveries(line_map):
    from medical_uav.env.render import outcome_text

    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])
    t0 = sim.t
    sim.do_deliver(2)
    text = outcome_text(sim, t0, sim.t)
    assert "1 delivered on time" in text and "battery" in text


def test_draw_with_highlight_probs_and_speed(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    r = Renderer(line_map, sim.cfg)
    plain = r.draw(sim, 0.0, "Decision 1")
    marked = r.draw(sim, 0.0, "Decision 1", highlight={0}, probs=[("PICK medicine M->H2", 0.8)],
                    speed="5 simulated min per second")
    assert marked.shape == plain.shape and not np.array_equal(marked, plain)


def test_outcome_and_highlight_exclude_events_at_the_start(line_map):
    from medical_uav.env.render import highlight_ids, outcome_text

    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])  # pickup event at t0 belongs to the previous decision
    t0 = sim.t
    sim.do_deliver(2)
    assert "picked up" not in outcome_text(sim, t0, sim.t)
    assert highlight_ids(sim, t0, sim.t) == {0}


def test_state_at_end_of_a_decision_shows_the_finished_action(line_map):
    a, b = make_order(0, 1, M, 2, 0.5, 0.0), make_order(1, 1, M, 3, 0.5, 0.0)
    sim = make_sim(line_map, [a, b])
    sim.do_pick(a)
    t1 = sim.t  # the load ends here and the next flight starts here
    sim.do_deliver(2)
    assert drone_state_at(sim, t1)["activity"] == "fly"
    done = drone_state_at(sim, t1, at_end=True)
    assert done["activity"] == "done" and done["segment"].kind == "load"
    assert [o.id for o in done["cargo"]] == [0]
