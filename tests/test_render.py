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
