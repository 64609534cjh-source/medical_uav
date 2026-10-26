"""Dev-only exact action/day validation, followed by isolated repeat CPU timing."""
import argparse,hashlib,json,time,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import torch
from medical_uav.agents.networks import build_network,masked_distribution
from medical_uav.agents.destination_inference import DestinationInferencePolicy,factored_forward
from medical_uav.agents.inference import ActionOnlyEvaluationPolicy
from medical_uav.evaluation.evaluate import evaluate
from medical_uav.training.trainer import load_checkpoint
from medical_uav.training.io import clean_json,write_json


def read(p):return json.loads(Path(p).read_text())


class CheckedPolicy(torch.nn.Module):
    def __init__(self,model,mode):
        super().__init__();self.model=model;self.policy=DestinationInferencePolicy(model) if mode=='factored' else ActionOnlyEvaluationPolicy(model);self.mode=mode;self.logit_error=0.;self.value_error=0.;self.decisions=0
    def act(self,obs,mask,deterministic=False):
        a=self.policy.act(obs,mask,deterministic)[0]
        logits,value=self.model(obs)
        if self.mode=='factored':
            other,w=factored_forward(self.model,obs);self.logit_error=max(self.logit_error,float((logits-other).abs().max()));self.value_error=max(self.value_error,float((value-w).abs().max()));torch.testing.assert_close(logits,other,atol=2e-6,rtol=1e-5);torch.testing.assert_close(value,w,atol=0,rtol=0)
        assert torch.equal(a,masked_distribution(logits,mask).probs.argmax(-1))
        self.decisions+=1
        return a,None,None


def worker(entry,output):
    if entry.get('checkpoint_sha256'):assert hashlib.sha256(Path(entry['checkpoint']).read_bytes()).hexdigest()==entry['checkpoint_sha256']
    torch.set_num_threads(1);state=load_checkpoint(entry['checkpoint']);model=build_network(state['architecture']);model.load_state_dict(state['model_state_dict'])
    raw=[json.loads(x) for x in (Path(entry['checkpoint']).parent/'evaluation_episodes.jsonl').read_text().splitlines()];original={(r['episode_load'],r['demand_seed']):{k:v for k,v in r.items() if k not in ('transitions','update_index')} for r in raw if r['policy']=='ppo' and r['transitions']==state['steps']};assert len(original)==120
    result={'name':entry['name'],'checkpoint':entry['checkpoint'],'modes':{}}
    with torch.inference_mode():
        for mode in ['argmax_only','factored']:
            policy=CheckedPolicy(model,mode);records,_=evaluate(policy,('ppo',),split='dev',learner_seed=state['config']['learner_seed'])
            for r in records:assert clean_json(r)==original[r['episode_load'],r['demand_seed']]
            result['modes'][mode]={'days':120,'decisions':policy.decisions,'all_actions_and_complete_day_fields_exact':True,'max_logit_abs_error':policy.logit_error,'max_value_abs_error':policy.value_error,'tie_fallback_states':getattr(policy.policy,'fallback_states',None)}
    write_json(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-entry',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--benchmark',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    if args.worker_entry:worker(read(args.worker_entry),args.output);return
    entries=read(args.protocol)['entries']
    if not args.benchmark:
        assert not args.output_dir.exists();args.output_dir.mkdir(parents=True)
        def run(e):
            entry=args.output_dir/(e['name']+'_entry.json');write_json(entry,e)
            with (args.output_dir/(e['name']+'.log')).open('w') as log:subprocess.run([sys.executable,__file__,'--worker-entry',str(entry),'--output',str(args.output_dir/(e['name']+'.json'))],stdout=log,stderr=subprocess.STDOUT,check=True)
            return e['name']
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(run,e) for e in entries]):print('Verified',f.result(),flush=True)
        write_json(args.output_dir/'manifest.json',{'status':'complete','protocol':read(args.protocol),'entries':len(entries)});return
    assert not args.output.exists();torch.set_num_threads(1)
    archive=np.load('outputs/m4_inference_benchmark/observations.npz');obs=torch.from_numpy(archive['obs']);mask=torch.from_numpy(archive['masks']);results=[]
    with torch.inference_mode():
        for e in entries:
            state=load_checkpoint(e['checkpoint']);model=build_network(state['architecture']);model.load_state_dict(state['model_state_dict']);model.eval()
            policies={'canonical':model,'argmax_only':ActionOnlyEvaluationPolicy(model).eval(),'factored_guarded':DestinationInferencePolicy(model).eval()};timings={}
            # Alternate variant order over three rounds to reduce order bias.
            for repeat in range(3):
                order=list(policies);order=order[repeat:]+order[:repeat]
                for name in order:
                    for batch in [1,8]:
                        policy=policies[name]
                        for _ in range(200):policy.act(obs[:batch],mask[:batch],True)
                        samples=[]
                        for i in range(2000):
                            start=(i*batch)%(len(obs)-batch+1);x=obs[start:start+batch];m=mask[start:start+batch];tick=time.perf_counter_ns();policy.act(x,m,True);samples.append((time.perf_counter_ns()-tick)/1000)
                        timings.setdefault(name,{}).setdefault(str(batch),[]).append({'p50_us_per_batch':float(np.median(samples)),'p95_us_per_batch':float(np.quantile(samples,.95))})
            results.append({**e,'parameters':sum(p.numel() for p in model.parameters()),'rounds':timings});print('Measured',e['name'],flush=True)
    aggregate={}
    for method in ['destination','urgency','coordinated']:
        subset=[r for r in results if r['method']==method];aggregate[method]={'parameters':subset[0]['parameters'],'variants':{}}
        for variant in ['canonical','argmax_only','factored_guarded']:
            aggregate[method]['variants'][variant]={}
            for batch in ['1','8']:
                aggregate[method]['variants'][variant][batch]={k:float(np.median([np.median([x[k] for x in r['rounds'][variant][batch]]) for r in subset])) for k in ['p50_us_per_batch','p95_us_per_batch']}
    write_json(args.output,{'threads':1,'device':'cpu','torch':str(torch.__version__),'scope':'isolated from project training/evaluation processes; identical archived real observations;3 round variant order rotation;200 warmups+2000 calls per round,model,variant,batch; inference_mode; medians of per-model round-median quantiles, not pooled p95','models':results,'aggregate':aggregate,'implementation':'original already vectorizes all6 destinations; factored_guarded reuses context in PICK and DELIVER, same underlying parameters, near-tie fallback1e-5; argmax_only is existing generic helper, observed Dev equivalence checked separately; training/scientific final evaluation remain canonical'});print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
