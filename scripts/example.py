"""Walk through the environment API: create it, inspect the state, act, render, run a full day.

Usage:
    python scripts/example.py
"""
import os

import imageio.v2 as imageio

from medical_uav.baselines.heuristics import run_episode, slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import action_name
from medical_uav.env.config import NODE_NAMES, ORDER_TYPES


def show_state(env):
    """Print the drone, the visible pending orders and the allowed actions."""
    sim = env.sim
    print(f"t = {sim.t:.1f} min | at {NODE_NAMES[sim.location]} | battery {sim.battery:.1f} % | "
          f"{len(sim.onboard)} box(es) on board")
    for k, o in enumerate(env.slots):
        print(f"  slot {k}: {ORDER_TYPES[o.type_idx].name:8s} {NODE_NAMES[o.origin]}->"
              f"{NODE_NAMES[o.dest]}  {o.weight:.2f} kg  deadline in {o.deadline - sim.t:.0f} min")
    allowed = [action_name(a) for a, ok in enumerate(env.action_masks()) if ok]
    print("  allowed:", ", ".join(allowed))


def main(frame_path="outputs/example_frame.png"):
    # 1) create the environment and start a day with a fixed demand seed
    env = MedicalUAVEnv(load="medium", render_mode="rgb_array")
    obs, info = env.reset(options={"demand_seed": 0})
    print(f"observation: {obs.shape[0]} numbers, actions: {env.action_space.n}")
    show_state(env)

    # 2) take a few decisions with the Slack-aware baseline and see what happens
    for _ in range(5):
        action = slack_aware_policy(env)
        print(f"\n> {env.describe(action)}")
        obs, reward, done, truncated, info = env.step(action)
        events = ", ".join(f"{kind} #{oid} at {t:.1f}" for t, kind, oid in info["events"])
        print(f"  took {info['elapsed']:.1f} min, reward {reward:+.3f}, events: {events or 'none'}")
        show_state(env)

    # 3) render the current state as an image
    os.makedirs(os.path.dirname(frame_path) or ".", exist_ok=True)
    imageio.imwrite(frame_path, env.render())
    print(f"\nframe saved to {frame_path}")

    # 4) run a whole day and print the main metrics
    m = run_episode(env, slack_aware_policy, demand_seed=0)
    print(f"full day: {m['n_orders']} orders, on time {m['on_time_rate']:.0%}, "
          f"expired {m['expiry_rate']:.0%}, return {m['return']:.2f}")


if __name__ == "__main__":
    main()
