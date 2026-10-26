"""Evaluate optional equivalent structured inference on development days only."""

import argparse
import hashlib
from pathlib import Path
import time

import torch

from medical_uav.agents.inference import FactoredStructuredInference, ActionOnlyEvaluationPolicy
from medical_uav.env.config import EnvConfig
from medical_uav.evaluation.evaluate import evaluate
from medical_uav.training.io import write_json
from medical_uav.training.trainer import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--split',choices=('dev','dev-extra','train'),default='dev')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--loads',nargs='+',choices=('low','medium','high'),default=['low','medium','high'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('choose a fresh output file')
    torch.set_num_threads(1)
    state=load_checkpoint(args.checkpoint)
    if state['architecture']!=FactoredStructuredInference.architecture:
        parser.error('a structured checkpoint is required')
    model=FactoredStructuredInference();model.load_state_dict(state['model_state_dict'])
    policy=ActionOnlyEvaluationPolicy(model).eval()
    with torch.inference_mode():
        tick=time.perf_counter()
        rows,summary=evaluate(policy,('ppo',),seeds=args.seeds,split=args.split,loads=args.loads,
                              learner_seed=state['config']['learner_seed'],env_config=EnvConfig(**state['env_config']))
        wall=time.perf_counter()-tick
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,{'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                           'path':'factored PICK + action-only masked argmax, near-tie historical fallback',
                           'wall_seconds':wall,'decisions':sum(r['decisions'] for r in rows),
                           'summary':summary,'episodes':rows})
    print(f'Saved {len(rows)} complete development episodes.')


if __name__=='__main__':main()
