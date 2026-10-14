"""Run one day with a baseline policy, print its metrics and show or save an animation.

Usage:
    python scripts/demo.py --live
    python scripts/demo.py --policy random --seed 1
    python scripts/demo.py --no-video --snapshots 0 100
"""
import argparse
import bisect
import math
import os
import sys

import imageio.v2 as imageio
import numpy as np

from medical_uav.baselines.heuristics import (edf_policy, random_policy, run_episode,
                                              slack_aware_policy)
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.render import Renderer, frame_times, save_animation
from medical_uav.metrics import episode_metrics

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "svg", "ps", "cairo", "template")


def make_policy(name: str, seed: int):
    """Return the baseline policy with this name."""
    if name == "slack":
        return slack_aware_policy
    if name == "edf":
        return edf_policy
    rng = np.random.default_rng(seed)
    return lambda env: random_policy(env, rng)


def has_display() -> bool:
    """Return True if a window can be shown on this machine."""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def print_metrics(metrics: dict) -> None:
    """Print the day's metrics, one per line."""
    for key, value in metrics.items():
        if isinstance(value, dict):
            value = "  ".join(f"{k} {v:.2f}" for k, v in value.items())
            print(f"  {key:15s} {value}")
        elif isinstance(value, float):
            print(f"  {key:15s} {value:.3f}")
        else:
            print(f"  {key:15s} {value}")


class Captions:
    """Frame captions: run settings and the action in progress."""

    def __init__(self, env, args):
        self.env = env
        self.prefix = f"{args.policy} | load {args.load} | demand seed {args.seed}"

    def at(self, t: float) -> str:
        log = self.env.decision_log
        # index of the last decision made at or before t; -1 if none yet
        i = bisect.bisect_right([d[0] for d in log], t + 1e-9) - 1
        action = f"{log[i][1]} (decided at t={log[i][0]:.0f})" if i >= 0 else "start"
        return f"{self.prefix} | t = {t:5.1f} min | {action}"


def live_window_problem() -> str | None:
    """Return why a live window cannot be opened, or None if it can."""
    if not has_display():
        return "no display found (e.g. remote session)"
    import matplotlib

    if matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS:
        return f"matplotlib backend {matplotlib.get_backend()} cannot open a window"
    return None


def run_live(env, policy, args) -> None:
    """Run the policy and show the day in a window as it unfolds."""
    import matplotlib.pyplot as plt

    renderer = Renderer(env.city_map, env.cfg)
    captions = Captions(env, args)
    env.reset(options={"demand_seed": args.seed})
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.canvas.manager.set_window_title("Medical UAV")
    ax.axis("off")
    image = ax.imshow(renderer.draw(env.sim, 0.0, captions.at(0.0)))

    done, t_prev = False, 0.0
    while not done and plt.fignum_exists(fig.number):
        _, _, done, _, _ = env.step(policy(env))
        for t in frame_times(t_prev, env.sim.t, args.minutes_per_frame):
            if not plt.fignum_exists(fig.number):
                break
            image.set_data(renderer.draw(env.sim, t, captions.at(t)))
            plt.pause(1.0 / args.fps)
        t_prev = env.sim.t

    if not done:
        print(f"partial run: window closed at t = {env.sim.t:.1f} min "
              "(metrics cover the orders released so far)")
    print_metrics(episode_metrics(env.sim))
    if done and plt.fignum_exists(fig.number):
        ax.set_title("day finished - close the window to exit", fontsize=12)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["slack", "edf", "random"], default="slack")
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
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if not (math.isfinite(args.minutes_per_frame) and args.minutes_per_frame > 0):
        parser.error("--minutes-per-frame must be a finite number > 0")
    if not all(math.isfinite(t) and t >= 0 for t in args.snapshots):
        parser.error("--snapshots must be finite times >= 0")
    if args.out is not None and not args.out.endswith((".gif", ".mp4")):
        parser.error("--out must end with .gif or .mp4")

    env = MedicalUAVEnv(load=args.load)
    policy = make_policy(args.policy, args.seed)
    print(f"policy={args.policy} load={args.load} demand_seed={args.seed} "
          f"map_seed={env.cfg.map_seed}")

    if args.live:
        problem = live_window_problem()
        if problem is None:
            run_live(env, policy, args)
            return
        print(f"{problem} - saving an animation instead")

    print_metrics(run_episode(env, policy, args.seed))
    renderer = Renderer(env.city_map, env.cfg)
    captions = Captions(env, args)

    for t in args.snapshots:
        t = min(t, env.sim.t)
        path = f"outputs/snapshot_{args.policy}_{args.load}_{args.seed}_t{t:.0f}.png"
        os.makedirs("outputs", exist_ok=True)
        imageio.imwrite(path, renderer.draw(env.sim, t, captions.at(t)))
        print(f"saved {path}")
    if args.no_video:
        return

    out = args.out or f"outputs/demo_{args.policy}_{args.load}_{args.seed}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    times = [0.0] + frame_times(0.0, env.sim.t, args.minutes_per_frame)
    frames = (renderer.draw(env.sim, t, captions.at(t)) for t in times)
    n = save_animation(frames, out, args.fps)
    print(f"saved {n} frames to {out}")


if __name__ == "__main__":
    main()
