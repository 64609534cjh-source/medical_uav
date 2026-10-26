"""Canonical CPU latency for all 15 preregistered best models; run in isolation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from medical_uav.agents.networks import build_network
from medical_uav.training.trainer import load_checkpoint
from medical_uav.training.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    torch.set_num_threads(1)
    first=json.loads(Path('configs/gen2_holdout_freeze.json').read_text())
    models=[{'name':e['name'],'method':'m5','learner_seed':e['learner_seed'],'checkpoint':e['checkpoint']} for e in first['reference_models']]
    models += [{'name':f'{m}_s{s}','method':m,'learner_seed':s,'checkpoint':f'outputs/gen2_architectures/{m}_s{s}/best.pt'} for m in ['destination','relational'] for s in range(1,6)]
    archive_path=Path('outputs/m4_inference_benchmark/observations.npz');archive=np.load(archive_path)
    obs=torch.from_numpy(archive['obs']);mask=torch.from_numpy(archive['masks']);results=[]
    with torch.inference_mode():
        for entry in models:
            checkpoint=Path(entry['checkpoint']);state=load_checkpoint(checkpoint);model=build_network(state['architecture']);model.load_state_dict(state['model_state_dict']);model.eval()
            values={}
            for batch in [1,8]:
                for _ in range(200):model.act(obs[:batch],mask[:batch],True)
                samples=[]
                for i in range(2000):
                    start=(i*batch)%(len(obs)-batch+1);x=obs[start:start+batch];m=mask[start:start+batch]
                    tick=time.perf_counter_ns();model.act(x,m,True);samples.append((time.perf_counter_ns()-tick)/1000)
                values[str(batch)]={'p50_us_per_batch':float(np.median(samples)),'p95_us_per_batch':float(np.quantile(samples,.95)),'warmups':200,'calls':2000}
            results.append({**entry,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parameters':sum(p.numel() for p in model.parameters()),'latency':values})
            print('Measured',entry['name'],flush=True)
    aggregate={}
    for method in ['m5','destination','relational']:
        subset=[r for r in results if r['method']==method]
        aggregate[method]={'parameters':subset[0]['parameters'],'by_batch':{}}
        for batch in ['1','8']:
            aggregate[method]['by_batch'][batch]={k:float(np.median([r['latency'][batch][k] for r in subset])) for k in ['p50_us_per_batch','p95_us_per_batch']}
    result={'threads':1,'device':'cpu','torch':str(torch.__version__),'observation_archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),'scope':'all 15 best models; identical real observation/mask archive; canonical masked categorical argmax including critic; tensor conversion excluded; isolated from project training/evaluation jobs; aggregate median of five per-model p50/p95 measurements; p95 is per-model, aggregate is not pooled tail quantile','models':results,'aggregate':aggregate}
    write_json(args.output,result);print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
