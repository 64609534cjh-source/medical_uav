"""Aggregate frozen observational R1 replays; no simulator or Test execution."""
import gzip,hashlib,json,statistics,tarfile,io,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LOADS=('low','medium','high'); TYPES=('plasma','medicine','sample','supplies')
def pct(n,d):return 100*n/d if d else None
def quantiles(values):return [float(x) for x in np.quantile(values,[.5,.95])] if values else None

def main():
 folder=ROOT/'outputs/gen4_r1_visibility';suite=json.loads((folder/'manifest.json').read_text())
 if suite['status']!='complete' or len(suite['completed'])!=10:raise RuntimeError('ten complete diagnostic replays required')
 result={'stage':'Gen4 R1 development-only diagnostic','analysis_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_commit':suite['source_commit'],'protocol_sha256':hashlib.sha256((ROOT/'configs/gen4_r1_protocol.json').read_bytes()).hexdigest(),'new_training_transitions':0,'new_test_evaluations':0,'methods':{},'same_state_groups':{},'representatives':[],'raw_files':{},'unchanged_original_days':0,'wall_seconds':suite['wall_seconds'],'scope':'Observational, decision-time visibility; no causal attribution. Pooled order percentages include five repetitions of shared demand days and are labelled separately from mean daily rates.'}
 lines=['# Gen4 R1：可见性、等待和动作机会诊断','','只回放原 Dev-extra（100 天 × 三档 × 五 learner × 两方法）。不重新选点、不训练、不读取新 Test。逐日结果和原诊断逐字段核对。','','| 方法 | 负载 | 订单类型 | 失败订单数* | 首次可见时已直接迟到 / 失败 % | 曾隐藏 / 失败 % | 曾有合法 PICK / 失败 % | 首次可见延迟中位/p95 分钟** |','|---|---|---|---:|---:|---:|---:|---|']
 data={}
 for method in ['m5','destination']:
  data[method]=[];result['methods'][method]={}
  for seed in range(1,6):
   path=folder/f'{method}_s{seed}.json.gz';result['raw_files'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
   payload=json.loads(gzip.decompress(path.read_bytes()));assert payload['source_commit']==suite['source_commit'];assert payload['protocol_sha256']==result['protocol_sha256'];assert payload['unchanged_archived_days']==300
   result['unchanged_original_days']+=payload['unchanged_archived_days'];data[method]+=payload['episodes']
  for load in LOADS:
   rows=[r for r in data[method] if r['episode_load']==load];orders=[o for r in rows for o in r['diagnostics']['visibility_audit']['orders']];failed=[o for o in orders if o['outcome']!='on_time']
   def summarize(group):
    failures=[o for o in group if o['outcome']!='on_time']
    wait=sum(o['pending_wait_minutes'] for o in group)
    return {'orders_repeated':len(group),'failures_repeated':len(failures),'failed_first_visible_direct_late':sum(o['first_visible_direct_slack_minutes'] is not None and o['first_visible_direct_slack_minutes']<0 for o in failures),'failed_ever_hidden':sum(o.get('hidden_decisions',0)>0 for o in failures),'failed_had_legal_pick':sum(o.get('legal_pick_decisions',0)>0 for o in failures),'failed_never_visible':sum(o['first_visible_delay_minutes'] is None for o in failures),'failed_first_visible_delay_p50_p95':quantiles([o['first_visible_delay_minutes'] for o in failures if o['first_visible_delay_minutes'] is not None]),'pending_wait_minutes':wait,'decision_held_visible_wait_minutes':sum(o.get('decision_held_visible_wait_minutes',0) for o in group),'decision_held_hidden_wait_minutes':sum(o.get('decision_held_hidden_wait_minutes',0) for o in group),'unobserved_between_decisions_wait_minutes':sum(o['unobserved_between_decisions_wait_minutes'] for o in group),'all_first_visible_delay_p50_p95':quantiles([o['first_visible_delay_minutes'] for o in group if o['first_visible_delay_minutes'] is not None]),'outcomes':{name:sum(o['outcome']==name for o in group) for name in ['on_time','late','expired_unpicked','expired_after_pick']}}
   summary=summarize(orders);summary['by_type']={typ:summarize([o for o in orders if o['type']==typ]) for typ in TYPES};summary['mean_daily_on_time_rate']=statistics.mean(r['on_time_rate'] for r in rows)
   result['methods'][method][load]=summary
   for typ,s in summary['by_type'].items():
    n=s['failures_repeated'];q=s['failed_first_visible_delay_p50_p95'];qtext='—' if q is None else f'{q[0]:.2f}/{q[1]:.2f}'
    def show(v):return '—' if v is None else f'{v:.2f}'
    lines.append('| '+' | '.join([method,load,typ,str(n),show(pct(s['failed_first_visible_direct_late'],n)),show(pct(s['failed_ever_hidden'],n)),show(pct(s['failed_had_legal_pick'],n)),qtext])+' |')
 lines+=['','*订单数重复同一需求日五次；不是独立需求样本。百分比是该条件下的 pooled 订单计数，不能替代总体主指标的日均率。**延迟为 release→first decision visibility；未可见的订单单独计数，不填零。','','## 等待时间分解','','| 方法 | 负载 | 已标记可见等待 % | 已标记隐藏等待 % | 到下一决策前未观测等待 % |','|---|---|---:|---:|---:|']
 for method in data:
  for load in LOADS:
   s=result['methods'][method][load];den=s['pending_wait_minutes'];lines.append('| '+' | '.join([method,load]+[f'{pct(s[k],den):.2f}' for k in ['decision_held_visible_wait_minutes','decision_held_hidden_wait_minutes','unobserved_between_decisions_wait_minutes']])+' |')
 lines+=['','标签仅在决策时刻观测，并保持至下一决策/取货/过期；不能称作宏动作内部连续真实候选占用。首次可见前延迟还可能来自充电/飞行/装卸期间没有新决策。','','## 同状态动作分歧','','每行的两个策略接收相同 observation/mask；访问状态来自表中轨迹策略。不同轨迹行不是同一状态分布。分组重叠，不应相加。','','| 轨迹策略 | 负载 | 状态组 | 状态数 | 动作不同 % | 轨迹策略 PICK % | 对方 PICK % | 轨迹策略 DELIVER % | 对方 DELIVER % |','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for method,rows in data.items():
  result['same_state_groups'][method]={}
  for load in LOADS:
   result['same_state_groups'][method][load]={}
   for group in ['all','overflow','urgent_live_cargo','battery_below_40']:
    counter={}
    for row in rows:
     if row['episode_load']!=load:continue
     for k,v in row['diagnostics']['visibility_audit']['state_groups'].get(group,{}).items():counter[k]=counter.get(k,0)+v
    result['same_state_groups'][method][load][group]=counter
    n=counter.get('states',0)
    if n:lines.append('| '+' | '.join([method,load,group,str(n)]+[f'{pct(counter.get(k,0),n):.2f}' for k in ['action_disagreements','chosen_pick','other_pick','chosen_deliver','other_deliver']])+' |')
 traces=json.loads(gzip.decompress((ROOT/'results/discussion_traces.json.gz').read_bytes()))['traces']
 lines+=['','## 四类代表轨迹的首次分歧','','| 类型 | 负载/日 | 首次分歧时刻 | M5 动作 | Destination 动作 | 最终日准时率差 pp |','|---|---|---:|---|---|---:|']
 for bucket in ['success','failure_both','improvement','regression']:
  pair={m:next(r for r in traces if r['method']==m and r['case']['bucket']==bucket) for m in data};first=None
  for a,b in zip(pair['m5']['steps'],pair['destination']['steps']):
   if a['action']!=b['action']:
    assert a['observation']==b['observation'] and a['mask']==b['mask'];first={'t':a['t'],'m5':a['action_name'],'destination':b['action_name'],'battery':a['battery'],'cargo_ids':a['cargo_ids']};break
  case=pair['m5']['case'];result['representatives'].append({'case':case,'first_divergence_common_state':first})
  lines.append(f'| {bucket} | {case["load"]}/{case["day"]} | '+(f'{first["t"]:.2f} | {first["m5"]} | {first["destination"]}' if first else '— | 相同 | 相同')+f' | {case["delta_pp"]:+.2f} |')
 lines+=['','轨迹按预先记录的成功/失败/改善/退化中位规则选择，只解释行为差异。首次动作分歧不是因果归责，也不是该动作价值的估计。','','## 下一步','','候选窗口需按实际隐藏和首次可见时机作有限对照；所有失败曾可见不能排除延迟或排序问题，但不能据此承诺 K=24 收益。短视规划需先通过公开状态重建、独立未来抽样和活环境不变性验收。合法 PICK 未选择并不证明应立即取货：容量、电量和已载货时限的机会成本仍在。','','新 Dev-tune 与 Dev-confirm 各 100 天已在碰撞审计后登记；本轮保持两者未评估，新 Test 未生成。确认集仅在候选配置冻结后开放并记录访问。保留原总体选点与所有类型取舍报告；0.5pp 仍是后续训练晋级规则的待冻结实践门槛。','','R1 核心诊断、共同场景登记和两项研究立项完成；独立组员 review、R3 学习训练及 R4 planner 性能验证不在本结论中。']
 (ROOT/'results/gen4_r1_summary.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'results/gen4_r1_summary.md').write_text('\n'.join(lines)+'\n')
 archive=ROOT/'artifacts/gen4_r1_visibility.tar.gz'
 with archive.open('wb') as stream,gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w') as tar:
   for path in sorted([folder/'manifest.json']+list(folder.glob('*.json.gz'))):
    content=path.read_bytes();entry=tarfile.TarInfo(path.relative_to(ROOT).as_posix());entry.size=len(content);entry.mode=0o644;tar.addfile(entry,io.BytesIO(content))
 (ROOT/'artifacts/gen4_r1_visibility_manifest.json').write_text(json.dumps({'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'members':result['raw_files'],'source_commit':suite['source_commit'],'scope':result['scope']},indent=2)+'\n')
 print(json.dumps({'unchanged_days':result['unchanged_original_days'],'raw_archive_bytes':archive.stat().st_size,'source_commit':suite['source_commit']}))
if __name__=='__main__':main()
