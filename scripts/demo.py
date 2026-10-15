"""Run one day with a baseline policy, print the results and save a frame and an animation."""
import argparse
import bisect
import os

import imageio.v2 as imageio
import numpy as np

from medical_uav.baselines.heuristics import (edf_policy, random_policy, run_episode,
                                              slack_aware_policy)
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.render import Renderer, clock, frame_times, save_animation
from medical_uav.metrics import summary_lines

POLICY_NAMES = {"slack": "Slack-aware", "edf": "EDF", "random": "Random"}
SNAPSHOT_MIN = 182.0  # minute of the saved frame; a busy moment of the default day
FPS = 12  # animation frames per second; one frame per simulated minute


def make_policy(name: str, seed: int):
    """Return the baseline policy with this name."""
    if name == "slack":
        return slack_aware_policy
    if name == "edf":
        return edf_policy
    rng = np.random.default_rng(seed)
    return lambda env: random_policy(env, rng)


def caption_at(env, t: float) -> str:
    """Return the line shown under the frame title: the last decision taken at or before t."""
    log = env.decision_log
    # index of the last decision made at or before t; -1 if none yet
    i = bisect.bisect_right([d[0] for d in log], t + 1e-9) - 1
    return f"Last decision {clock(log[i][0])}: {log[i][1]}" if i >= 0 else "Start of the day"


def main():
    parser = argparse.ArgumentParser(description="Run one day with a baseline policy.")
    parser.add_argument("--policy", choices=list(POLICY_NAMES), default="slack",
                        help="slack = Slack-aware, edf = earliest deadline first, random")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="demand seed (which day of orders)")
    args = parser.parse_args()

    env = MedicalUAVEnv(load=args.load)
    name = POLICY_NAMES[args.policy]
    print(f"Medical UAV | {name} | {args.load} load | demand seed {args.seed} | "
          f"map seed {env.cfg.map_seed}")

    metrics = run_episode(env, make_policy(args.policy, args.seed), args.seed)
    sim = env.sim
    print(f"\nCompleted at {clock(sim.t)} (t={sim.t:.1f} min) "
          f"after {metrics['decisions']} decisions")
    for line in summary_lines(metrics):
        print("  " + line)
    print()

    renderer = Renderer(env.city_map, env.cfg,
                        title=f"Medical UAV  |  {name}  |  {args.load} load  |  "
                              f"demand seed {args.seed}")
    base = f"outputs/demo_{args.policy}_{args.load}_seed{args.seed}"
    os.makedirs("outputs", exist_ok=True)

    t = min(SNAPSHOT_MIN, sim.t)
    imageio.imwrite(f"{base}_t{round(t, 1):g}min.png", renderer.draw(sim, t, caption_at(env, t)))
    print(f"Saved: {base}_t{round(t, 1):g}min.png")

    times = [0.0] + frame_times(0.0, sim.t, 1.0)
    print(f"Rendering {len(times)} frames at {FPS} fps "
          f"(about {len(times) / FPS:.0f} s of animation)...")
    save_animation((renderer.draw(sim, t, caption_at(env, t)) for t in times), f"{base}.gif", FPS)
    print(f"Saved: {base}.gif")


if __name__ == "__main__":
    main()
