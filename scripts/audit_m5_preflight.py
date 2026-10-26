import hashlib,json
from pathlib import Path
extra=range(100040,100140);used=[];files=0

def inspect(value,path):
    if isinstance(value,dict):
        if isinstance(value.get('demand_seed'),int) and value['demand_seed'] in extra:
            used.append({'file':str(path),'demand_seed':value['demand_seed']})
        for v in value.values():
            if isinstance(v,(dict,list)):inspect(v,path)
    elif isinstance(value,list):
        for v in value:
            if isinstance(v,(dict,list)):inspect(v,path)
for path in Path('outputs').rglob('*'):
    if path.suffix not in ('.json','.jsonl') or 'm5_audit' in path.parts:continue
    files+=1
    if path.suffix=='.jsonl':
        with path.open() as stream:
            for line in stream:inspect(json.loads(line),path)
    else:inspect(json.loads(path.read_text()),path)
result={'checked_json_and_jsonl_files':files,'extra_range':[100040,100139],'previous_occurrences':used,'scope':'all existing local outputs, recorded demand_seed fields; no assertion about unrecorded external experiments','m4_commit':'46ce56d','frozen_results':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('results').glob('m4*') if p.is_file()},'frozen_m3_result_sha256':hashlib.sha256(Path('results/m3_mlp_ppo.json').read_bytes()).hexdigest()}
assert not used,used[:10]
Path('results/m5_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print('Audited',files,'files; Dev-extra unused in recorded local experiments.')
