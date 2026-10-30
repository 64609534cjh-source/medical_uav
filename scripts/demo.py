"""Run one day with a policy, print the results and save a replay, a frame and a video.

This is also the usage example: create the environment, reset it to a day, step until the day
is over, then read the results and show the day.

Outputs, in outputs/:
  *_replay.html    the whole day in a web browser: play, pause, step through the decisions
  *_decision.png   one decision worth a look, as in the replay
  *_backlog.mp4    with --video: the backlog around the opening decision, each decision slowed down
                   as in the replay
"""
import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.heuristics import edf_policy, random_policy, slack_aware_policy
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.render import clock, save_animation
from medical_uav.env.replay import (day_record, decision_record, opening_decision, state_at,
                                    write_html)
from medical_uav.env.video import ReplayFrames, explained_times
from medical_uav.metrics import episode_metrics, summary_lines

POLICY_NAMES = {"slack": "Slack-aware rule", "edf": "EDF rule", "random": "Random policy",
                "mlp": "MLP trained by imitation", "set": "Set network trained by imitation",
                "ppo": "PPO agent"}
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"
CHECKPOINT_FILES = {"mlp": "bc_mlp.pt", "set": "bc_set.pt", "ppo": "ppo_set.pt"}


def make_policy(name: str, seed: int, checkpoint: str | None = None):
    """Return the named policy, called as policy(env, obs, mask), and its network if any; a
    network policy loads `checkpoint` if given."""
    if name in CHECKPOINT_FILES:
        net, _ = load_checkpoint(checkpoint or CHECKPOINTS / CHECKPOINT_FILES[name])
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
                             "mlp / set = the networks trained by imitation, ppo = the PPO "
                             "agent (all in checkpoints/)")
    parser.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="demand seed (which day of orders)")
    parser.add_argument("--map", type=int, default=0,
                        help="map seed (0: the map of the results; e.g. 2000-2029: test maps)")
    parser.add_argument("--checkpoint", default=None,
                        help="network file for mlp/set/ppo, e.g. one saved by train_ppo.py")
    parser.add_argument("--video", action="store_true",
                        help="also save a video of the backlog (a few minutes)")
    parser.add_argument("--fps", type=int, default=12, help="video frames per second")
    args = parser.parse_args()

    name = POLICY_NAMES[args.policy] + (f" ({Path(args.checkpoint).name})" if args.checkpoint
                                        else "")
    policy, network = make_policy(args.policy, args.seed, args.checkpoint)

    # 1. create the environment and start a day
    env = MedicalUAVEnv(config=EnvConfig(map_seed=args.map), load=args.load)
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
    base = (f"outputs/demo_{args.policy}_{args.load}_seed{args.seed}"
            + (f"_map{args.map}" if args.map else ""))
    os.makedirs("outputs", exist_ok=True)
    record = day_record(env, decisions, name)
    write_html(record, f"{base}_replay.html")
    print(f"Saved: {base}_replay.html (open it in a web browser)")

    # 5. a frame of one decision worth a look, and the videos, drawn from the same record
    renderer = ReplayFrames(record, f"{args.load.capitalize()} load · demand seed {args.seed} · "
                                    f"map {env.cfg.map_seed}")
    first = opening_decision(record)
    still = renderer.draw(record["decisions"][first]["t"], "Chosen")
    imageio.imwrite(f"{base}_decision.png", still)
    print(f"Saved: {base}_decision.png")
    if not args.video:
        print("Add --video to also save a video of the backlog (a few minutes).")
        return
    # the backlog: from where 3 or more orders wait in the opening decision's demand peak until
    # fewer than 3 wait for the last time
    decs = record["decisions"]
    waiting = [len(state_at(record, d["t"])["waiting"]) for d in decs]
    peaks = [p for p in record["meta"]["peaks"] if p[0] <= decs[first]["t"]]
    start = next((i for i, d in enumerate(decs) if peaks and d["t"] >= peaks[-1][0]
                  and waiting[i] >= 3), first)
    busy = [i for i in range(start, len(decs)) if waiting[i] >= 3]
    stop = min(busy[-1] + 1, len(decs) - 1) if busy else start
    print(f"Drawing the backlog video, decisions {start + 1}-{stop + 1} ...", flush=True)
    times = explained_times(record, start, stop - start + 1, args.fps)
    note = "the replay at 1× · pauses for reading, not decision time"
    n = save_animation(frames(renderer, times, note), f"{base}_backlog.mp4", args.fps)
    print(f"Saved: {base}_backlog.mp4 ({n / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
