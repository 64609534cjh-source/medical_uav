"""Summarize both bounded engineering probes, including null results and cost."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOADS = ('low', 'medium', 'high')
VARIANTS = ('reference', 'rollout_no_value', 'rollout_value')
VERSIONS = {'v1': 'gen4_lookahead_feasibility', 'v2': 'gen4_lookahead_clone_v2'}


def main():
    output = {'scope': 'Engineering feasibility: one preselected frozen learner, five '
              'predetermined Dev-tune days x three loads, two implementation versions. '
              'No multi-seed method improvement claim; latency includes fallback/cold calls.',
              'analysis_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                               cwd=ROOT, text=True).strip(),
              'versions': {}, 'raw_files': {}, 'new_training_transitions': 0,
              'new_confirmation_episodes': 0, 'new_test_episodes': 0}
    output['compute_scope'] = ('Simulator counts cover the two registered 45-day probes; '
                               'unit tests and the single Train-state micro-profile are '
                               'separate engineering checks excluded from these counts.')
    lines = ['# Gen4 R4：公开信息短视规划的计算可行性', '', output['scope'], '',
             '协议先于评估冻结：M=4、H=30 模拟分钟、20ms 软预算；真实未来订单与 demand seed '
             '不进入 planner。v2 只改合成分支复制，并逐动作验证无预算时的分数和物理状态等价。', '',
             '| 版本 | 策略 | Low % | Medium % | High % | 等权 % | reward 日均 | 推理 p50/p95/max ms |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    raw_members = []
    known_reference = None
    for version, folder_name in VERSIONS.items():
        folder = ROOT / 'outputs' / folder_name
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'complete':
            raise RuntimeError('incomplete feasibility experiment')
        raw_members.append(manifest_path)
        output['versions'][version] = {'source_commit': manifest['source_commit'],
                                       'protocol_sha256': manifest['protocol_sha256'],
                                       'wall_seconds': manifest['wall_seconds'],
                                       'cpu': manifest['cpu'], 'threads': manifest['threads'],
                                       'methods': {}}
        reference_rows = None
        for variant in VARIANTS:
            path = folder / (variant+'.json.gz')
            raw_members.append(path)
            output['raw_files'][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            data = json.loads(gzip.decompress(path.read_bytes()))
            rows = data['episodes']
            keys = {(r['episode_load'], r['demand_seed']) for r in rows}
            expected = {(load, day) for load in LOADS for day in range(100140, 100145)}
            if len(rows) != 15 or keys != expected or any(
                    not r['complete_day'] or r['invalid_actions'] or r['split'] != 'dev-tune' for r in rows):
                raise RuntimeError('invalid/duplicate/incomplete feasibility days')
            keyed = {(r['episode_load'], r['demand_seed']): r for r in rows}
            if variant == 'reference':
                reference_rows = keyed
                if known_reference is not None and reference_rows != known_reference:
                    raise RuntimeError('reference day fields changed between implementation probes')
                known_reference = reference_rows
            means = [statistics.mean(r['on_time_rate'] for r in rows if r['episode_load'] == load)
                     for load in LOADS]
            counts = data['planning']['counts'] if data['planning'] else {}
            result = {'mean_on_time_by_load': dict(zip(LOADS, means)),
                      'equal_on_time_rate': statistics.mean(means),
                      'mean_reward': statistics.mean(r['reward'] for r in rows),
                      'mean_served_rate': statistics.mean(r['served_rate'] for r in rows),
                      'mean_expiry_rate': statistics.mean(r['expiry_rate'] for r in rows),
                      'latency_ms_p50_p95_max': data['latency_ms_p50_p95_max'],
                      'planning_counts': counts, 'decisions': sum(r['decisions'] for r in rows),
                      'days_differing_in_on_time': sum(r['on_time_rate'] != reference_rows[key]['on_time_rate']
                                                     for key, r in keyed.items()),
                      'per_type_daily_on_time': {}}
            for kind in ('plasma', 'medicine', 'sample', 'supplies'):
                values = [r['on_time_by_type'][kind] for r in rows if r['on_time_by_type'][kind] is not None]
                result['per_type_daily_on_time'][kind] = {'eligible_days': len(values),
                                                         'mean': statistics.mean(values) if values else None}
            output['versions'][version]['methods'][variant] = result
            latency = '/'.join(f'{v:.3f}' for v in result['latency_ms_p50_p95_max'])
            lines.append('| '+' | '.join([version, variant]+[f'{100*v:.2f}' for v in means]
                         + [f'{100*result["equal_on_time_rate"]:.2f}', f'{result["mean_reward"]:.3f}', latency])+' |')
    lines += ['', '同一版本只使用一个预先固定的 learner；±跨种子标准差和确认性置信区间均未估计。'
              '这里的 87.79% 属于五个开发日，不能与旧 Test-v3 的 86.00% 相减为“提升”。', '',
              '## 实际覆盖与额外计算', '',
              '| 版本 | 规划 | 全部决策 | 信息足够状态 | 完成比较 | 完成/全部 % | 超时回退 | 改变动作 | 额外模拟 transitions |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    total_sim = 0
    for version in VERSIONS:
        for variant in VARIANTS[1:]:
            c = output['versions'][version]['methods'][variant]['planning_counts']
            total_sim += c.get('extra_simulator_transitions', 0)
            completed = c.get('completed_comparisons', 0)
            lines.append(f'| {version} | {variant} | {c["decisions"]} | {c["eligible_states"]} | '
                         f'{completed} | {100*completed/c["decisions"]:.2f} | {c.get("fallback_budget", 0)} | '
                         f'{c.get("changed_actions", 0)} | {c.get("extra_simulator_transitions", 0)} |')
    output['extra_simulator_transitions'] = total_sim
    output['complete_evaluation_days_including_repeated_controls'] = 90
    output['unique_demand_day_ids'] = 5
    lines += ['', '历史已经完成的订单，其释放时间戳不在当前 403 维观测中。当前原型拒绝无法还原最近到达历史的状态，'
              '每个规划版本有 695/1,765 次因此回退；隐藏待取订单 32 次，无已释放工作 466 次。'
              '这些是互斥回退原因，不能把全部回退归因于速度。', '',
              '所有方法在每一个完整日的准时率均与原策略相同；四个规划版本各仅改变 4 次动作。'
              '这支持“当前 20ms 原型覆盖太低、未观察到服务收益”，不能推出规划一般无效。'
              'v2 完成比较从 16/17 次增至 19 次，仍只占约 1%，不构成实质工程晋级。', '',
              '延迟是同一 CPU、单线程、顺序执行的完整决策计时，包含回退和冷启动首调用。'
              '20ms 是软预算，按宏模拟动作结束检查；最大观测超时仍完整保留，没有硬实时保证。'
              '这不是重复多轮的稳定微基准，不解读小幅计时差异为确定的加速率。', '',
              f'合计 {total_sim:,} 额外模拟 transitions，90 个完整评估日（含重复原策略），只有 5 个独立需求日 ID、'
              '一个冻结 learner；真实训练 transitions 为 0。这些计数只覆盖两轮已登记评估，'
              '不含单状态 Train profiling 和单元测试的工程开销。No-value/value-tail 的额外模拟量必须与训练样本分开核算。', '',
              '## 本轮决定与下一步', '',
              '保留目的地网络与已验收快路径为正式默认，不因这组小样本改选方法、seed 或奖励。'
              '当前规划原型不进入大规模性能确认；先验证批量候选/情景推理能否提高固定预算下的完整比较覆盖。'
              '若加入公开历史记忆以重建到达历史，必须作为单独注册的状态/信息变化，不能偷偷读取 live simulator。', '',
              'Dev-tune 只使用 100140–100144；Dev-confirm 及其余新开发日保持未评估，新 Final-Test 未生成。'
              '两个工程版本及其空结果全部保留；没有扩大 M/H/时间预算，也没有额外训练搜索。', '',
              '![Planning coverage and end-to-end latency](gen4_lookahead_feasibility.png)', '',
              '逐日所有类型指标和费用在 JSON/压缩原始记录中。原始数据归档及成员 hash 由独立 manifest 索引；'
              '原始 source SHA 对应各自 Git 历史，不能用 v2 文件冒充 v1 的冻结源码。']
    if any(m['days_differing_in_on_time'] for v in output['versions'].values() for m in v['methods'].values()):
        raise RuntimeError('update narrative: some complete-day on-time rates changed')
    (ROOT / 'results/gen4_lookahead_feasibility.json').write_text(json.dumps(output, indent=2)+'\n')
    (ROOT / 'results/gen4_lookahead_feasibility.md').write_text('\n'.join(lines)+'\n')
    figure(output)
    archive = ROOT / 'artifacts/gen4_lookahead_feasibility.tar.gz'
    raw_members += [ROOT / 'outputs/gen4_lookahead_feasibility/copy_profile.json',
                    ROOT / 'outputs/gen4_lookahead_feasibility/profile.txt']
    with archive.open('wb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in sorted(raw_members):
                content = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                info.size, info.mode = len(content), 0o644
                tar.addfile(info, io.BytesIO(content))
    index = {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
             'members': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_members}}
    (ROOT / 'artifacts/gen4_lookahead_feasibility_manifest.json').write_text(json.dumps(index, indent=2)+'\n')
    access = {'scope': 'Development access only; no new Test generated',
              'dev_tune_ids_used': list(range(100140, 100145)),
              'dev_tune_ids_remaining_unevaluated': list(range(100145, 100240)),
              'dev_confirm_accesses': [], 'new_final_test': None,
              'experiments': [{'version': v, 'source_commit': x['source_commit'],
                              'variants': list(x['methods']), 'episodes': 45}
                             for v, x in output['versions'].items()],
              'training_transitions': 0, 'planning_simulator_transitions': total_sim}
    (ROOT / 'results/gen4_development_access.json').write_text(json.dumps(access, indent=2)+'\n')
    print(json.dumps({'complete_days': 90, 'extra_simulator_transitions': total_sim,
                      'archive_bytes': archive.stat().st_size, 'confirmation_accesses': 0}))


def figure(output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.25, 1]})
    labels, values = [], []
    categories = ['No released work', 'Missing public history', 'Hidden orders', 'Budget fallback', 'Completed']
    colors = ['#dddddd', '#a8b6c6', '#54759a', '#e6b968', '#b35e18']
    for version in VERSIONS:
        for variant in VARIANTS[1:]:
            c = output['versions'][version]['methods'][variant]['planning_counts']
            labels.append(version+' '+('no tail' if variant == 'rollout_no_value' else 'value tail'))
            values.append([100*c.get(k, 0)/c['decisions'] for k in ('fallback_no_released_work',
                           'fallback_unrepresented_recent_arrival_history', 'fallback_hidden_pending_orders',
                           'fallback_budget', 'completed_comparisons')])
    left = np.zeros(len(labels))
    for i, (category, color) in enumerate(zip(categories, colors)):
        amounts = np.array(values)[:, i]
        axes[0].barh(labels, amounts, left=left, color=color, label=category, height=.6)
        left += amounts
    axes[0].set_xlim(0, 100)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Share of live decisions (%)')
    axes[0].set_title('Planning coverage and fallback reasons', loc='left')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncols=2, frameon=False, fontsize=9)
    names, latency = [], []
    for version in VERSIONS:
        for variant in VARIANTS:
            names.append(version+' '+{'reference': 'reference', 'rollout_no_value': 'no tail',
                                     'rollout_value': 'value tail'}[variant])
            latency.append(output['versions'][version]['methods'][variant]['latency_ms_p50_p95_max'])
    for i, (name, data) in enumerate(zip(names, latency)):
        axes[1].plot([data[0], data[2]], [i, i], color='#cccccc', lw=1)
    for j, (marker, label, color) in enumerate([('o', 'p50', '#54759a'), ('D', 'p95', '#b35e18'),
                                                ('|', 'maximum', '#555555')]):
        axes[1].scatter(np.array(latency)[:, j], range(len(names)), marker=marker,
                        color=color, s=30, label=label, zorder=3)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].axvline(20, color='#777777', ls='--', lw=1)
    axes[1].set_xlim(0, 50)
    axes[1].set_xlabel('End-to-end policy decision latency (ms)')
    axes[1].set_title('Observed latency; 20ms soft budget', loc='left')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncols=3, frameon=False)
    fig.suptitle('Public-information rollout: bounded engineering feasibility', x=.07, ha='left', fontsize=15)
    fig.text(.07, .02, 'One frozen learner; five Dev-tune days x three loads; one sequential pass per version. '
             'Quantiles describe latency, not confidence intervals.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.1, right=.98, top=.85, bottom=.3, wspace=.65)
    for suffix in ('png', 'svg'):
        path = ROOT / f'results/gen4_lookahead_feasibility.{suffix}'
        fig.savefig(path, dpi=160, facecolor='white')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
