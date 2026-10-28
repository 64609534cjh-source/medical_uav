"""Run one day with a policy, print the results and save a replay, a frame and two videos.

This is also the usage example: create the environment, reset it to a day, step until the day
is over, then read the results and show the day.

Outputs, in outputs/:
  *_replay.html    the whole day in a web browser: play, pause, step through the decisions
  *_decision.png   one decision worth a look, as in the replay
  *_explained.mp4  with --video: the demand peak around the opening decision, each decision held so it can be read
  *_day.mp4        with --video: the whole day at the replay's smart speed
"""
import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.heuristics import edf_policy, random_policy, slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.render import clock, save_animation
from medical_uav.env.replay import (day_record, decision_record, opening_decision, state_at,
                                    write_html)
from medical_uav.env.video import ReplayFrames, explained_times, smart_times
from medical_uav.metrics import episode_metrics, summary_lines

POLICY_NAMES = {"slack": "Slack-aware rule", "edf": "EDF rule", "random": "Random policy",
                "mlp": "MLP trained by imitation", "set": "Set network trained by imitation"}
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


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


def frames(renderer: ReplayFrames, times, speed: str):
    """Frames for (time, phase) pairs; a held frame is drawn once and repeated."""
    key, frame = None, None
    for t, phase in times:
        if (t, phase) != key:
            key, frame = (t, phase), renderer.draw(t, phase, speed)
        yield frame


def main():
    parser = argparse.ArgumentParser(description="Run one day with a policy.")
    parser.add_argument("--policy", choices=list(POLICY_NAMES), default="slack",
                        help="slack = Slack-aware, edf = earliest deadline first, random, "
                             "mlp / set = the networks trained by imitation (checkpoints/)")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="demand seed (which day of orders)")
    parser.add_argument("--video", action="store_true",
                        help="also save the explanation and day videos (15-20 minutes)")
    parser.add_argument("--explain-count", type=int, default=31,
                        help="decisions in the explanation video, which starts where orders "
                             "pile up (3 or more waiting) in the peak of the opening decision")
    parser.add_argument("--fps", type=int, default=12, help="video frames per second")
    args = parser.parse_args()

    name = POLICY_NAMES[args.policy]
    policy, network = make_policy(args.policy, args.seed)

    # 1. create the environment and start a day
    env = MedicalUAVEnv(load=args.load)
    obs, info = env.reset(options={"demand_seed": args.seed})
    print(f"Medical UAV | {name} | {args.load} load | demand seed {args.seed} | "
          f"map seed {env.cfg.map_seed}")

    # 2. let the policy choose actions until the day is over; rule-based policies read the
    #    environment, a network uses only obs and info["action_mask"]; each decision is also
    #    recorded (before the step) for the replay
    total_reward, decisions = 0.0, []
    terminated = truncated = False
    while not (terminated or truncated):
        probs = network.probabilities(obs, info["action_mask"]) if network else None
        action = policy(env, obs, info["action_mask"])
        decisions.append(decision_record(env, action, probs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    # 3. results of the day
    sim = env.sim
    metrics = episode_metrics(sim)
    metrics["return"] = total_reward
    print(f"\nCompleted at {clock(sim.t)} (t={sim.t:.1f} min) after {len(decisions)} decisions")
    for line in summary_lines(metrics):
        print("  " + line)
    print()

    # 4. the replay: one HTML file with the whole day, opened in any web browser
    base = f"outputs/demo_{args.policy}_{args.load}_seed{args.seed}"
    os.makedirs("outputs", exist_ok=True)
    record = day_record(env, decisions, name)
    write_html(record, f"{base}_replay.html")
    print(f"Saved: {base}_replay.html (open it in a web browser)")

    # 5. a frame of one decision worth a look, and the videos, drawn from the same record
    renderer = ReplayFrames(record, f"{args.load.capitalize()} load · demand seed {args.seed} · "
                                    f"map {env.cfg.map_seed}")
    first = opening_decision(record)
    imageio.imwrite(f"{base}_decision.png", renderer.draw(record["decisions"][first]["t"]))
    print(f"Saved: {base}_decision.png")
    if not args.video:
        print("Add --video to also save the explanation and day videos (15-20 minutes).")
        return
    print("Drawing the explanation video (a few minutes) ...", flush=True)
    decs = record["decisions"]
    peaks = [p for p in record["meta"]["peaks"] if p[0] <= decs[first]["t"]]
    start = next((i for i, d in enumerate(decs) if peaks and d["t"] >= peaks[-1][0]
                  and len(state_at(record, d["t"])["waiting"]) >= 3), first)
    times = explained_times(record, start, args.explain_count, args.fps)
    note = "slowed down for reading · the network decides in about 1 ms"
    n = save_animation(frames(renderer, times, note),
                       f"{base}_explained.mp4", args.fps)
    print(f"Saved: {base}_explained.mp4 ({n / args.fps:.0f} s)")
    print("Drawing the day video (about 10 minutes) ...", flush=True)
    times = [(t, None) for t in smart_times(record, args.fps)]
    n = save_animation(frames(renderer, times, "smart speed: flights 1, other 8 simulated "
                                               "min/s · pauses for reading"),
                       f"{base}_day.mp4", args.fps)
    print(f"Saved: {base}_day.mp4 ({n / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
