"""Evaluate a saved neural policy and/or same-day baselines; Test is explicit only."""

import argparse
from pathlib import Path

import torch

from medical_uav.agents.networks import build_network
from medical_uav.env.config import EnvConfig
from medical_uav.evaluation.evaluate import evaluate
from medical_uav.training.io import isolated_rng, write_json
from medical_uav.training.trainer import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policies", nargs="+", default=["ppo", "random", "edf", "slack"],
                        choices=("ppo", "random", "edf", "slack"))
    parser.add_argument("--split", choices=("train", "dev", "dev-extra", "test", "test-v2"), default="dev")
    parser.add_argument("--seeds", nargs="+", type=int, help="optional subset within the selected split")
    parser.add_argument("--loads", nargs="+", choices=("low", "medium", "high"), default=["low", "medium", "high"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--congestion", action="store_true", help="read-only top-12 instrumentation")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--learner-seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    if "ppo" in args.policies and not args.checkpoint:
        parser.error("--checkpoint is required for PPO")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    network, config, seed = None, EnvConfig(), args.learner_seed
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint)
        config = EnvConfig(**state["env_config"])
        seed = state["config"]["learner_seed"]
        with isolated_rng():
            network = build_network(state["architecture"])
        network.load_state_dict(state["model_state_dict"])
    from medical_uav.evaluation.congestion import CongestionDiagnostics
    records, summary = evaluate(network, args.policies, args.seeds, args.split,
                                args.loads, seed, env_config=config, diagnostics=args.diagnostics or args.congestion,
                                diagnostic_factory=CongestionDiagnostics if args.congestion else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"checkpoint": str(args.checkpoint) if args.checkpoint else None,
                             "split": args.split, "summary": summary, "episodes": records})
    print(f"Saved {len(records)} complete evaluation days to {args.output}")


if __name__ == "__main__":
    main()
