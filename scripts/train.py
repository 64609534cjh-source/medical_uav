"""Continuous masked PPO; all step budgets count transitions across every environment."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from medical_uav.training.trainer import TrainConfig, load_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON TrainConfig, including nested ppo settings")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh directory for this run")
    parser.add_argument("--resume", type=Path, help="update-boundary checkpoint; start new training days")
    parser.add_argument("--architecture", choices=("mlp", "structured", "destination", "relational"))
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--learner-seed", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--loads", nargs="+", choices=("low", "medium", "high"))
    parser.add_argument("--train-seeds", nargs="+", type=int, help="optional repeated Train days for debugging")
    parser.add_argument("--eval-split", choices=("train", "dev"))
    parser.add_argument("--eval-interval-updates", type=int)
    args = parser.parse_args()
    if args.resume and args.config:
        parser.error("use the checkpoint config when resuming; do not also pass --config")
    data = (load_checkpoint(args.resume)["config"] if args.resume else
            json.loads(args.config.read_text()) if args.config else asdict(TrainConfig()))
    for key in ("architecture", "total_steps", "learner_seed", "threads", "num_envs", "rollout_steps", "loads",
                "train_seeds", "eval_split", "eval_interval_updates"):
        if getattr(args, key) is not None:
            data[key] = getattr(args, key)
    result = train(TrainConfig.from_dict(data), args.output_dir, args.resume)
    print(json.dumps({k: result[k] for k in ("status", "transitions", "updates", "wall_seconds", "best_score")}, indent=2))


if __name__ == "__main__":
    main()
