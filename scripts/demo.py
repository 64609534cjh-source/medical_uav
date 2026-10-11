"""Run one day with a baseline policy, print its metrics and show or save an animation.

Examples:
    python scripts/demo.py --policy edf --load medium --seed 0 --live   # play in a window
    python scripts/demo.py --policy random --seed 1                      # save a gif
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np

from medical_uav.baselines.heuristics import edf_policy, random_policy, run_episode
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.render import Renderer, frame_times, save_animation
from medical_uav.metrics import episode_metrics


def has_display() -> bool:
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def print_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        print(f"  {key:15s} {value:.3f}" if isinstance(value, float) else f"  {key:15s} {value}")


def run_live(env, policy, demand_seed: int, minutes_per_frame: float, fps: int) -> dict:
    """Step the policy and play the drone's movement in a window as the day unfolds."""
    import matplotlib.pyplot as plt

    renderer = Renderer(env.city_map, env.cfg)
    env.reset(options={"demand_seed": demand_seed})
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.canvas.manager.set_window_title("Medical UAV")
    ax.axis("off")
    image = ax.imshow(renderer.draw(env.sim, 0.0))

    done, t_prev = False, 0.0
    while not done and plt.fignum_exists(fig.number):
        _, reward, done, _, info = env.step(policy(env))
        ax.set_title(f"{info['action_name']}   reward {reward:+.2f}", fontsize=12)
        for t in frame_times(t_prev, env.sim.t, minutes_per_frame):
            image.set_data(renderer.draw(env.sim, t))
            plt.pause(1.0 / fps)
            if not plt.fignum_exists(fig.number):
                break
        t_prev = env.sim.t

    metrics = episode_metrics(env.sim)
    if plt.fignum_exists(fig.number):
        ax.set_title("day finished - close the window to exit", fontsize=12)
        plt.show()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "edf"], default="edf")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="demand seed (which day of orders)")
    parser.add_argument("--live", action="store_true", help="play in a window while running")
    parser.add_argument("--out", default=None, help="output .gif or .mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--minutes-per-frame", type=float, default=1.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--snapshots", type=float, nargs="*", default=[],
                        help="also save PNG frames at these times (minutes)")
    args = parser.parse_args()

    env = MedicalUAVEnv(load=args.load)
    if args.policy == "edf":
        policy = edf_policy
    else:
        rng = np.random.default_rng(args.seed)
        policy = lambda e: random_policy(e, rng)  # noqa: E731
    print(f"policy={args.policy} load={args.load} demand_seed={args.seed} "
          f"map_seed={env.cfg.map_seed}")

    if args.live:
        if has_display():
            print_metrics(run_live(env, policy, args.seed, args.minutes_per_frame, args.fps))
            return
        print("no display found (e.g. remote session) - saving an animation instead")

    print_metrics(run_episode(env, policy, args.seed))

    renderer = Renderer(env.city_map, env.cfg)
    for t in args.snapshots:
        path = f"outputs/snapshot_{args.policy}_{args.load}_{args.seed}_t{t:.0f}.png"
        os.makedirs("outputs", exist_ok=True)
        imageio.imwrite(path, renderer.draw(env.sim, min(t, env.sim.t)))
        print(f"saved {path}")
    if args.no_video:
        return
    out = args.out or f"outputs/demo_{args.policy}_{args.load}_{args.seed}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    times = [0.0] + frame_times(0.0, env.sim.t, args.minutes_per_frame)
    frames = [renderer.draw(env.sim, t) for t in times]
    save_animation(frames, out, args.fps)
    print(f"saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
