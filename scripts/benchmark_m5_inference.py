"""Verify inference reuse on fixed Dev policies, then measure isolated CPU latency."""
import argparse,json,time
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from medical_uav.agents.networks import build_network,StructuredActorCritic
from medical_uav.agents.inference import FactoredStructuredInference,masked_argmax
from medical_uav.evaluation.evaluate import evaluate
from medical_uav.training.trainer import load_checkpoint
from medical_uav.training.io import write_json

class EvaluationAdapter(nn.Module):
    """Private action-only evaluator; never used by collector or PPO updater."""
    def __init__(self,model,reference=None):
        super().__init__();self.model=model;self.reference=reference
        self.logit_max_error=0.;self.value_max_error=0.;self.checked_decisions=0
    def act(self,obs,mask,deterministic=False):
        assert deterministic
        action=masked_argmax(self.model,obs,mask)
        if self.reference is not None:
            a,v=self.reference(obs);b,w=self.model(obs)
            self.logit_max_error=max(self.logit_max_error,float((a-b).abs().max()))
            self.value_max_error=max(self.value_max_error,float((v-w).abs().max()))
            assert torch.equal(action,self.reference.act(obs,mask,True)[0])
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
            self.checked_decisions+=1
        return action,None,None


def records(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]


def model_at(path,fast=False):
    state=load_checkpoint(path)
    model=FactoredStructuredInference() if fast else build_network(state['architecture'])
    model.load_state_dict(state['model_state_dict']);return model.eval(),state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate-root',type=Path,required=True,help='root with seed_1..seed_5 or a JSON seed-to-run mapping');parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('fresh output required')
    args.output_dir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    mapping=json.loads(args.candidate_root.read_text()) if args.candidate_root.suffix=='.json' else {str(s):str(args.candidate_root/f'seed_{s}') for s in range(1,6)}
    paths=[]
    for s in range(1,6):
        for choice in ('best','last'):paths.append((f'm4_{choice}_s{s}',ROOT/f'outputs/m4_structured_ppo/seed_{s}',choice))
        if Path(mapping[str(s)]).resolve()!=(ROOT/f'outputs/m4_structured_ppo/seed_{s}').resolve():paths.append((f'candidate_best_s{s}',Path(mapping[str(s)]),'best'))
    verified=[]
    with torch.inference_mode():
        for name,run,choice in paths:
            reference,state=model_at(run/f'{choice}.pt');fast,_=model_at(run/f'{choice}.pt',True);adapter=EvaluationAdapter(fast,reference).eval()
            actual,_=evaluate(adapter,('ppo',),split='dev',learner_seed=state['config']['learner_seed'])
            original={(r['episode_load'],r['demand_seed']):r for r in records(run/'evaluation_episodes.jsonl') if r['policy']=='ppo' and r['transitions']==state['steps']}
            for r in actual:
                old=original[(r['episode_load'],r['demand_seed'])]
                assert r=={k:v for k,v in old.items() if k not in ('transitions','update_index')}
            verified.append({'model':name,'days':120,'decisions':adapter.checked_decisions,'logit_max_abs_error':adapter.logit_max_error,'value_max_abs_error':adapter.value_max_error,'all_actions_and_day_fields_exact':True})
            print('Verified',name,flush=True)
        archive=np.load(ROOT/'outputs/m4_inference_benchmark/observations.npz');obs=torch.from_numpy(archive['obs']);mask=torch.from_numpy(archive['masks'])
        selected=Path(mapping['1'])/'best.pt';reference,state=model_at(selected);factored,_=model_at(selected,True)
        mlp,_=model_at(ROOT/'outputs/m3_mixed_mlp_ppo/seed_1/best.pt')
        methods={'mlp_original':lambda x,m:mlp.act(x,m,True),
                 'structured_original':lambda x,m:reference.act(x,m,True),
                 'structured_argmax_only':lambda x,m:masked_argmax(reference,x,m),
                 'structured_factored_argmax':lambda x,m:masked_argmax(factored,x,m)}
        latency={}
        for name,call in methods.items():
            latency[name]={}
            for batch in (1,8):
                for _ in range(200):call(obs[:batch],mask[:batch])
                samples=[]
                for i in range(2000):
                    start=(i*batch)%(len(obs)-batch+1);x=obs[start:start+batch];m=mask[start:start+batch]
                    tick=time.perf_counter_ns();call(x,m);samples.append((time.perf_counter_ns()-tick)/1000)
                latency[name][str(batch)]={'p50_us_per_batch':float(np.median(samples)),'p95_us_per_batch':float(np.quantile(samples,.95)),
                                          'states_per_second_at_p50':batch*1e6/float(np.median(samples)),'calls':len(samples),'warmups':200}
        wall={};day_counts={}
        for name,policy in [('original',reference),('factored_argmax',EvaluationAdapter(factored).eval())]:
            samples=[];counts=[]
            for _ in range(3):
                tick=time.perf_counter();rs,_=evaluate(policy,('ppo',),split='dev',learner_seed=1);samples.append(time.perf_counter()-tick);counts.append(sum(r['decisions'] for r in rs))
            assert len(set(counts))==1
            wall[name]={'seconds':samples,'median_seconds':median(samples),'episodes':120,'decisions':counts[0]}
        assert wall['original']['decisions']==wall['factored_argmax']['decisions']
    result={'threads':1,'device':'cpu','torch':str(torch.__version__),'scope':'isolated; no learner/evaluation suite concurrent; same archived real observation inputs',
            'parameters':{'mlp':sum(p.numel() for p in mlp.parameters()),'structured':sum(p.numel() for p in reference.parameters())},
            'linear_multiply_accumulates':{'mlp':139392,'structured_original':267584,'structured_factored':177472,'reuse_saved':90112},
            'validated_models':verified,'validated_days':sum(v['days'] for v in verified),'latency':latency,'full_dev':wall,
            'implementation':'optional inference subclass; original state_dict keys and all training forwards unchanged; reference fallback for legal-logit gaps <=1e-5',
            'logits_real_arithmetic_equivalent':True,'floating_point_bitwise_logits_claimed':False,'selected_candidate_checkpoint':str(selected)}
    write_json(args.output_dir/'benchmark.json',result);print(json.dumps({'latency':latency,'full_dev':wall},indent=2),flush=True)

if __name__=='__main__':main()
