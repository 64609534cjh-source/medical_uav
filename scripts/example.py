"""Walk through the environment API: create it, inspect the state, act, render, run whole days.

Usage:
    python scripts/example.py
"""
import os

import imageio.v2 as imageio
import numpy as np

from medical_uav.baselines.heuristics import (edf_policy, random_policy, run_episode,
                                              slack_aware_policy)
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.actions import action_name
from medical_uav.env.config import NODE_NAMES, ORDER_TYPES
from medical_uav.env.render import clock
from medical_uav.metrics import share, summary_lines

SEED = 0  # demand seed: which day of orders


def show_state(env, allowed: bool = False) -> None:
    """Print the drone, the visible pending orders and, if asked, the allowed actions."""
    sim, cfg = env.sim, env.cfg
    print(f"  {clock(sim.t)} (t={sim.t:.1f} min) | at {NODE_NAMES[sim.location]} | "
          f"battery {sim.battery:.1f}% | cargo {len(sim.onboard)}/{cfg.max_boxes} boxes, "
          f"{sim.load_kg():.2f}/{cfg.max_weight:.2f} kg | pending {len(sim.pending)}")
    for k, o in enumerate(env.slots):
        print(f"  slot {k}: #{o.id} {ORDER_TYPES[o.type_idx].name} "
              f"{NODE_NAMES[o.origin]}->{NODE_NAMES[o.dest]}, {o.weight:.2f} kg, "
              f"due in {o.deadline - sim.t:.0f} min, expires in {o.expiry - sim.t:.0f} min")
    if allowed:
        names = [action_name(a) for a in np.flatnonzero(env.action_masks())]
        print("  allowed:", ", ".join(names))


def act(env, action: int, allowed: bool = False) -> None:
    """Run one action and print how long it took, its reward and what happened."""
    print(f"> {env.describe(action)}")
    _, reward, _, _, info = env.step(action)
    events = ", ".join(f"{kind} #{oid} at t={t:.1f}" for t, kind, oid in info["events"])
    print(f"  took {info['elapsed']:.1f} min | reward {reward:+.3f} | {events or 'no events'}")
    show_state(env, allowed)


def main(frame_path: str = "outputs/example_frame.png") -> None:
    print("1. Create the environment and start a day")
    env = MedicalUAVEnv(load="medium", render_mode="rgb_array")
    obs, _ = env.reset(options={"demand_seed": SEED})
    print(f"  medium load, demand seed {SEED}, map seed {env.cfg.map_seed}")
    print(f"  observation: {obs.shape[0]} numbers | actions: {env.action_space.n}")
    show_state(env, allowed=True)

    print("\n2. Let the Slack-aware baseline take five decisions")
    for k in range(5):
        act(env, slack_aware_policy(env), allowed=k == 0)

    print("\n3. Choose an action ourselves: charge to 100 %")
    allowed = np.flatnonzero(env.action_masks())
    act(env, int(next(a for a in allowed if action_name(a).endswith(",100)"))))

    print("\n4. Render the current state")
    os.makedirs(os.path.dirname(frame_path) or ".", exist_ok=True)
    imageio.imwrite(frame_path, env.render())
    print(f"  Saved: {frame_path}")

    print("\n5. Run the whole day again from 08:00 with Slack-aware")
    m = run_episode(env, slack_aware_policy, SEED)
    print(f"  full day: finished at {clock(env.sim.t)} after {m['decisions']} decisions")
    for line in summary_lines(m):
        print("  " + line)

    print(f"\n6. Compare the baselines on the same day ({m['n_orders']} orders)")
    rng = np.random.default_rng(SEED)
    policies = {"Random": lambda e: random_policy(e, rng), "EDF": edf_policy,
                "Slack-aware": slack_aware_policy}
    print(f"  {'policy':12s} {'on time':>14s} {'late':>5s} {'expired':>8s} {'reward':>8s}")
    for name, policy in policies.items():
        r = run_episode(env, policy, SEED)
        n, expired = r["n_orders"], round(r["expiry_rate"] * r["n_orders"])
        print(f"  {name:12s} {share(r['on_time'], n):>14s} {r['delivered'] - r['on_time']:5d} "
              f"{expired:8d} {r['return']:8.2f}")

    print("\n7. Change a setting: the drone carries one box instead of three")
    small = MedicalUAVEnv(config=EnvConfig(max_boxes=1), load="medium")
    r = run_episode(small, slack_aware_policy, SEED)
    print(f"  Slack-aware, max_boxes=1: on time {share(r['on_time'], r['n_orders'])}, "
          f"reward {r['return']:.2f} (three boxes: {share(m['on_time'], m['n_orders'])})")


if __name__ == "__main__":
    main()
