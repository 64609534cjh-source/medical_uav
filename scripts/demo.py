"""Run one day with a policy, print the results and save a frame and two videos.

This is also the usage example: create the environment, reset it to a day, step until the day
is over, then read the results and draw the day.

Videos:
  *_explained.mp4  a few consecutive decisions, each held on screen so it can be read
  *_day.mp4        the whole day at a labelled speed
"""
import argparse
import bisect
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.heuristics import edf_policy, random_policy, slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.render import (Renderer, clock, explanation_times, highlight_ids, outcome_text,
                                    overview_times, save_animation)
from medical_uav.metrics import episode_metrics, summary_lines

POLICY_NAMES = {"slack": "Slack-aware", "edf": "EDF", "random": "Random",
                "mlp": "BC MLP network", "set": "BC set network"}
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"
SNAPSHOT_MIN = 182.0  # minute of the saved frame and of the first explained decision
FPS = 12  # video frames per second
HOLD_S, ACTION_S = 2.0, 3.0  # seconds a decision is held before and after, and its action lasts


def make_policy(name: str, seed: int):
    """Return the named policy, called as policy(env, obs, mask), and its network if any."""
    if name in ("mlp", "set"):
        net, _ = load_checkpoint(CHECKPOINTS / f"bc_{name}.pt")
        return NetworkPolicy(net), NetworkPolicy(net)
    if name == "random":
        rng = np.random.default_rng(seed)
        return (lambda env, obs, mask: random_policy(env, rng)), None
    rule = slack_aware_policy if name == "slack" else edf_policy
    return (lambda env, obs, mask: rule(env)), None


def top_choices(env, network: NetworkPolicy, obs, mask) -> list[tuple[str, float]]:
    """The network's three most likely actions, in words, with their probabilities."""
    p = network.probabilities(obs, mask)
    return [(env.describe(int(a)), float(p[a])) for a in np.argsort(p)[::-1][:3] if p[a] >= 0.005]


def decision_at(env, t: float) -> int:
    """Index of the last decision taken at or before t; -1 before the first one."""
    return bisect.bisect_right([d[0] for d in env.decision_log], t + 1e-9) - 1


def caption_at(env, t: float) -> str:
    """Return the line shown under the frame title: the last decision taken at or before t."""
    i = decision_at(env, t)
    log = env.decision_log
    return f"Last decision {clock(log[i][0])}: {log[i][1]}" if i >= 0 else "Start of the day"


def main():
    parser = argparse.ArgumentParser(description="Run one day with a policy.")
    parser.add_argument("--policy", choices=list(POLICY_NAMES), default="slack",
                        help="slack = Slack-aware, edf = earliest deadline first, random, "
                             "mlp / set = the trained networks in checkpoints/")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="demand seed (which day of orders)")
    parser.add_argument("--explain-count", type=int, default=8,
                        help="decisions shown in the explanation video")
    parser.add_argument("--day-speed", type=float, default=5.0,
                        help="simulated minutes per second in the day video")
    args = parser.parse_args()

    name = POLICY_NAMES[args.policy]
    policy, network = make_policy(args.policy, args.seed)

    # 1. create the environment and start a day
    env = MedicalUAVEnv(load=args.load)
    obs, info = env.reset(options={"demand_seed": args.seed})
    print(f"Medical UAV | {name} | {args.load} load | demand seed {args.seed} | "
          f"map seed {env.cfg.map_seed}")

    # 2. let the policy choose actions until the day is over; rule-based policies read the
    #    environment, a network uses only obs and info["action_mask"]
    total_reward, choices = 0.0, []
    terminated = truncated = False
    while not (terminated or truncated):
        if network is not None:
            choices.append(top_choices(env, network, obs, info["action_mask"]))
        action = policy(env, obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    # 3. results of the day
    sim = env.sim
    metrics = episode_metrics(sim)
    metrics["return"] = total_reward
    log = env.decision_log
    print(f"\nCompleted at {clock(sim.t)} (t={sim.t:.1f} min) after {len(log)} decisions")
    for line in summary_lines(metrics):
        print("  " + line)
    print()

    # 4. draw a frame and two videos; the renderer can draw any minute of the finished day
    renderer = Renderer(env.city_map, env.cfg,
                        title=f"Medical UAV  |  {name}  |  {args.load} load  |  "
                              f"demand seed {args.seed}")
    base = f"outputs/demo_{args.policy}_{args.load}_seed{args.seed}"
    os.makedirs("outputs", exist_ok=True)

    t = min(SNAPSHOT_MIN, sim.t)
    imageio.imwrite(f"{base}_t{round(t, 1):g}min.png", renderer.draw(sim, t, caption_at(env, t)))
    print(f"Saved: {base}_t{round(t, 1):g}min.png")

    # explanation: consecutive decisions from SNAPSHOT_MIN, each held, its orders highlighted
    first = max(decision_at(env, t), 0)
    ends = [d[0] for d in log[1:]] + [sim.t]  # when each decision's action finished
    frames = explanation_times(log, first, args.explain_count, HOLD_S, ACTION_S, FPS, sim.t)

    def explained(t, caption):
        i = int(re.search(r"Decision (\d+)", caption).group(1)) - 1
        t0, t1 = log[i][0], ends[i]
        done = "done" in caption
        if done:
            caption += f"  ->  {outcome_text(sim, t0, t1)}"
        return renderer.draw(sim, t, caption, highlight=highlight_ids(sim, t0, t1),
                             probs=choices[i] if choices else None, speed="decision by decision",
                             at_end=done)

    n = save_animation((explained(t, c) for t, c in frames), f"{base}_explained.mp4", FPS)
    print(f"Saved: {base}_explained.mp4 ({n / FPS:.0f} s)")

    # the whole day at a labelled speed
    speed = f"{args.day_speed:g} simulated min per second"
    times = overview_times(sim.t, args.day_speed, FPS)
    def overview(t):
        probs = choices[max(decision_at(env, t), 0)] if choices else None
        return renderer.draw(sim, t, caption_at(env, t), speed=speed, probs=probs)

    n = save_animation((overview(t) for t in times), f"{base}_day.mp4", FPS)
    print(f"Saved: {base}_day.mp4 ({n / FPS:.0f} s)")


if __name__ == "__main__":
    main()
