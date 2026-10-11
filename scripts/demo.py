"""Run one episode with a baseline policy, print its metrics and save an animation.

Example:
    python scripts/demo.py --policy edf --load medium --seed 0
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np

from medical_uav.baselines.heuristics import edf_policy, random_policy, run_episode
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.render import Renderer, save_animation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "edf"], default="edf")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0)
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

    metrics = run_episode(env, policy, seed=args.seed)
    print(f"policy={args.policy} load={args.load} seed={args.seed} map_seed={env.cfg.map_seed}")
    for key, value in metrics.items():
        print(f"  {key:15s} {value:.3f}" if isinstance(value, float) else f"  {key:15s} {value}")

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
    times = np.arange(0.0, env.sim.t + 1e-9, args.minutes_per_frame)
    frames = [renderer.draw(env.sim, t) for t in times]
    save_animation(frames, out, args.fps)
    print(f"saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
