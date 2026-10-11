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


def test_frame_times_cover_interval_without_repeating_start():
    from medical_uav.env.render import frame_times

    assert frame_times(0.0, 3.0, 1.0) == pytest.approx([1.0, 2.0, 3.0])
    assert frame_times(2.5, 4.0, 1.0) == pytest.approx([3.0, 4.0])
    assert frame_times(5.0, 5.0, 1.0) == []


def test_save_animation_streams_generator(tmp_path):
    import imageio.v2 as imageio

    from medical_uav.env.render import save_animation

    # distinct frames: GIF encoders merge identical consecutive frames
    frames = (np.full((32, 48, 3), 40 * i, dtype=np.uint8) for i in range(5))
    assert save_animation(frames, str(tmp_path / "a.gif"), fps=5) == 5
    assert len(imageio.mimread(tmp_path / "a.gif")) == 5


def test_draw_accepts_header(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    frame = Renderer(line_map, sim.cfg).draw(sim, 0.0, header="slack | load medium | seed 0")
    assert frame.ndim == 3


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
