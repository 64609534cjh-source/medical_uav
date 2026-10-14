"""Watch or record one day with a baseline policy and print its results.

Usage:
    python scripts/demo.py --live
    python scripts/demo.py --policy edf --load high --seed 3
    python scripts/demo.py --no-video --snapshots 60 222
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
from medical_uav.env.render import FRAME_PX, Renderer, clock, frame_times, save_animation
from medical_uav.metrics import episode_metrics, summary_lines

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "svg", "ps", "cairo", "template")
POLICY_NAMES = {"slack": "Slack-aware", "edf": "EDF", "random": "Random"}
DEFAULT_SNAPSHOT = 166.0  # minutes; a busy moment of the default day


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


class Captions:
    """The decision shown under the frame title: the last one taken at or before t."""

    def __init__(self, env):
        self.env = env

    def at(self, t: float) -> str:
        log = self.env.decision_log
        # index of the last decision made at or before t; -1 if none yet
        i = bisect.bisect_right([d[0] for d in log], t + 1e-9) - 1
        return f"Last decision {clock(log[i][0])}: {log[i][1]}" if i >= 0 else "Start of the day"


def print_results(env, metrics: dict, decisions: int, partial: bool = False) -> None:
    """Print when the day ended and its main metrics."""
    sim = env.sim
    if partial:
        print(f"\nPartial run: stopped at {clock(sim.t)} (t={sim.t:.1f} min); "
              "metrics cover the orders released so far")
    else:
        print(f"\nCompleted at {clock(sim.t)} (t={sim.t:.1f} min) after {decisions} decisions")
    for line in summary_lines(metrics):
        print("  " + line)


def save_snapshots(env, renderer, captions, times, base: str) -> None:
    """Save a PNG frame at each requested time, clamped to the end of the run."""
    for t in times:
        if t > env.sim.t:
            print(f"t={t:g} min is after the end of the run; using t={env.sim.t:.1f}")
            t = env.sim.t
        path = f"{base}_t{round(t, 1):g}min.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, renderer.draw(env.sim, t, captions.at(t)))
        print(f"Saved: {path}")


def live_window_problem() -> str | None:
    """Return why a live window cannot be opened, or None if it can."""
    if not has_display():
        return "no display found (e.g. remote session)"
    import matplotlib

    if matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS:
        return f"matplotlib backend {matplotlib.get_backend()} cannot open a window"
    return None


def run_live(env, policy, renderer, captions, args) -> bool | None:
    """Show the day in a window as it unfolds.

    Return True if the day finished, False if the window was closed early, and None if no
    window could be opened.
    """
    import matplotlib.pyplot as plt

    env.reset(options={"demand_seed": args.seed})
    try:
        fig, ax = plt.subplots(figsize=(FRAME_PX[0] / 80, FRAME_PX[1] / 80))
        fig.canvas.manager.set_window_title("Medical UAV")
    except Exception as exc:  # the window system failed although a display was detected
        print(f"could not open a window ({exc})")
        return None
    fig.subplots_adjust(0, 0, 1, 1)
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

    metrics = episode_metrics(env.sim)
    metrics["return"] = env.sim.total_reward
    print_results(env, metrics, len(env.decision_log), partial=not done)
    if done and plt.fignum_exists(fig.number):
        print("Day finished: close the window to exit")
        plt.show()
    return done


def main():
    parser = argparse.ArgumentParser(description="Watch or record one day with a baseline.")
    parser.add_argument("--policy", choices=list(POLICY_NAMES), default="slack",
                        help="slack = Slack-aware, edf = earliest deadline first, random")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="demand seed (which day of orders)")
    parser.add_argument("--live", action="store_true", help="play in a window while running")
    parser.add_argument("--out", default=None, help="animation file, .gif or .mp4")
    parser.add_argument("--no-video", action="store_true", help="do not save an animation")
    parser.add_argument("--snapshots", type=float, nargs="*", default=[DEFAULT_SNAPSHOT],
                        help="save PNG frames at these minutes (none: no PNG)")
    parser.add_argument("--fps", type=int, default=12, help="animation frames per second")
    parser.add_argument("--minutes-per-frame", type=float, default=1.0,
                        help="simulated minutes between animation frames")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if not (math.isfinite(args.minutes_per_frame) and args.minutes_per_frame > 0):
        parser.error("--minutes-per-frame must be a finite number > 0")
    if not all(math.isfinite(t) and t >= 0 for t in args.snapshots):
        parser.error("--snapshots must be finite times >= 0")
    if args.out is not None and not args.out.endswith((".gif", ".mp4")):
        parser.error("--out must end with .gif or .mp4")
    if args.out is not None and args.no_video:
        parser.error("--out and --no-video cannot be used together")
    if args.live and (args.out is not None or args.no_video):
        parser.error("--live shows the day in a window; it cannot be combined with --out or "
                     "--no-video")

    env = MedicalUAVEnv(load=args.load)
    policy = make_policy(args.policy, args.seed)
    name = POLICY_NAMES[args.policy]
    print(f"Medical UAV | {name} | {args.load} load | demand seed {args.seed} | "
          f"map seed {env.cfg.map_seed}")
    renderer = Renderer(env.city_map, env.cfg,
                        title=f"Medical UAV  |  {name}  |  {args.load} load  |  "
                              f"demand seed {args.seed}")
    captions = Captions(env)
    base = f"outputs/demo_{args.policy}_{args.load}_seed{args.seed}"

    if args.live:
        problem = live_window_problem()
        if problem is None:
            finished = run_live(env, policy, renderer, captions, args)
            if finished:
                save_snapshots(env, renderer, captions, args.snapshots, base)
            if finished is not None:
                return
            problem = "no window"
        print(f"{problem}: saving an animation instead")

    metrics = run_episode(env, policy, args.seed)
    print_results(env, metrics, metrics["decisions"])
    print()
    save_snapshots(env, renderer, captions, args.snapshots, base)
    if args.no_video:
        return

    out = args.out or f"{base}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    times = [0.0] + frame_times(0.0, env.sim.t, args.minutes_per_frame)
    print(f"Rendering {len(times)} frames at {args.fps} fps "
          f"(about {len(times) / args.fps:.0f} s of animation)...")
    frames = (renderer.draw(env.sim, t, captions.at(t)) for t in times)
    save_animation(frames, out, args.fps)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
