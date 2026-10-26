"""Audit M4 against untouched M3 and repeat matched-day behavioral diagnostics.

All uncertainty is reported over five learner repetitions; daily differences are
retained separately and are descriptive selected-Dev evidence, not Test results.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

from scripts.analyze_m3 import analyze, read, records, day_index, digest
from scripts.diagnose_m4 import replay, aggregate_behavior
from medical_uav.agents.networks import build_network
from medical_uav.env import MedicalUAVEnv
from medical_uav.splits import DEV_SEEDS
from medical_uav.training.io import append_json, write_json
from medical_uav.training.trainer import load_checkpoint

LOADS = ("low", "medium", "high")
SEEDS = (1, 2, 3, 4, 5)
STEPS = tuple(range(0, 1_024_001, 102_400))


def daily_at(run, selection):
    state = load_checkpoint(run / f"{selection}.pt")
    index = day_index([r for r in records(run / "evaluation_episodes.jsonl")
                       if r["policy"] == "ppo" and r["transitions"] == state["steps"]])
    return state, index


def stats(values):
    return {"mean": mean(values), "sample_sd_across_seeds": stdev(values), "values": values}


def service_table(group):
    fields = ("on_time_rate", "reward", "on_time", "late_deliveries", "expired", "energy_used",
              "served_rate", "expiry_rate", "charge_minutes", "wait_minutes", "km_flown")
    result = {}
    for load in LOADS:
        selected = [r for r in group if r["episode_load"] == load]
        result[load] = {k: mean(r[k] for r in selected) for k in fields}
        result[load]["action_minutes_per_day"] = {kind: mean(r["action_minutes"].get(kind, 0) for r in selected)
                                                 for kind in ("pick", "deliver", "charge", "return", "wait")}
        result[load]["type_daily_rates"] = {}
        for type_name in selected[0]["type_counts"]:
            type_rows = [r["type_counts"][type_name] for r in selected if r["type_counts"][type_name]["released"]]
            result[load]["type_daily_rates"][type_name] = {
                "nonempty_days": len(type_rows),
                **{k: mean(r[k] / r["released"] for r in type_rows) for k in ("on_time", "delivered", "expired", "late_deliveries")}}
    return result


def charts(audit, mlp, pairs, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#225E9A", "#AF7621", "#42866A", "#9A4D8B", "#C66144")
    for offset in (0, 1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for ax, load in zip(axes.flat, (*LOADS, "mean")):
            def rates(curve):
                return [r["mean_on_time_rate"] if load == "mean" else r["by_load"][load]["on_time_rate"] for r in curve][offset:]
            ys = []
            for seed, color in zip(SEEDS, colors):
                y = rates(audit["curves"][str(seed)]); ys.append(y)
                ax.plot(np.array(STEPS[offset:]) / 1000, y, color=color, alpha=.5, lw=1, label=f"Structured seed {seed}")
            ax.plot(np.array(STEPS[offset:]) / 1000, np.mean(ys, axis=0), color="#143C62", lw=2.6, label="Structured mean")
            mlp_y = np.mean([rates(mlp["curves"][str(seed)]) for seed in SEEDS], axis=0)
            ax.plot(np.array(STEPS[offset:]) / 1000, mlp_y,
                    color="#AD650D", lw=2.4, ls="--", label="MLP mean")
            slack = audit["baselines"]["slack"]
            value = slack["mean_on_time_rate"] if load == "mean" else slack["by_load"][load]["on_time_rate"]
            ax.axhline(value, color="#3F7554", ls=":", lw=1.8, label="Slack-aware")
            ax.set(title=load.title() if load != "mean" else "Equal-load mean", xlabel="Training transitions (thousands)", ylabel="Dev mean daily on-time rate")
            ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.set_ylim(0, 1.02) if offset == 0 else ax.set_ylim(bottom=max(0, min(min(min(y) for y in ys), min(mlp_y), value) - .02), top=min(1.002, max(max(max(y) for y in ys), max(mlp_y), value) + .02))
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle("M4: five independent structured learners vs frozen MLP\nSame Dev days and transition budget; checkpoint curves use current weights", fontsize=14)
        for suffix in ("png", "svg"): fig.savefig(str(output) + ("_curves." if offset == 0 else "_tail_curves.") + suffix, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, load in zip(axes, LOADS):
        groups = [[100*r["delta_structured_mlp"] for r in pairs if r["selection"] == "best" and r["load"] == load and r["learner_seed"] == seed] for seed in SEEDS]
        ax.boxplot(groups, tick_labels=[str(s) for s in SEEDS], showfliers=True,
                   medianprops={"color": "#174E78"}, flierprops={"markersize": 2, "alpha": .35})
        ax.axhline(0, color="#555555", lw=1)
        ax.scatter(range(1, 6), [mean(g) for g in groups], color="#C05A21", marker="D", s=24, label="Mean of 40 days", zorder=3)
        ax.set(title=load.title(), xlabel="Learner seed", ylabel="Structured − MLP (percentage points)"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Best vs best on matched Dev days\nDay distributions are descriptive; learner uncertainty uses five seed means", fontsize=13)
    for suffix in ("png", "svg"): fig.savefig(str(output) + "_paired." + suffix, dpi=180)
    plt.close(fig)


def report(result, audit, mlp, path):
    lines = ["# M4: shared-order encoder vs frozen flat MLP", "", "All results below use the same reused Dev demand days. Test remains unused. Best checkpoints were independently selected by the frozen equal-load on-time rule, with daily reward as the tie-breaker. No learner seed was selected or dropped.", "", "## Selected checkpoints", "", "| Seed | Best step | Low % | Medium % | High % | Equal % | MLP equal % | Paired change (pp) | Last equal % |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(audit["best_per_seed"]):
        values=[100*row["by_load"][load]["on_time_rate"] for load in LOADS]
        lines.append(f"| {i+1} | {row['transitions']:,} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {100*row['mean_on_time_rate']:.2f} | {100*mlp['best_per_seed'][i]['mean_on_time_rate']:.2f} | {100*result['seed_deltas']['best']['mean']['values'][i]:+.2f} | {100*audit['last_per_seed'][i]['mean_on_time_rate']:.2f} |")
    lines += ["", "## Across five learner seeds", "", "Mean ± sample SD (ddof=1); these are not confidence intervals. Matching seed numbers organize repetitions but do not imply identical trajectories across architectures.", "", "| Model / selection | Low % | Medium % | High % | Equal % |", "|---|---:|---:|---:|---:|"]
    for name, source, selection in (("MLP best",mlp,"best"),("Structured best",audit,"best"),("MLP last",mlp,"last"),("Structured last",audit,"last")):
        vals=source[f"{selection}_across_seed_stats"]
        formatted=[f"{100*vals[k]['mean']:.2f} ± {100*vals[k]['sample_sd']:.2f}" for k in (*LOADS,"mean")]
        lines.append('| '+name+' | '+' | '.join(formatted)+' |')
    for name in ("slack", "edf", "random"):
        r=audit["baselines"][name];values=[r["by_load"][k]["on_time_rate"] for k in LOADS]+[r["mean_on_time_rate"]]
        lines.append('| '+name+' | '+' | '.join(f'{100*x:.2f}' for x in values)+' |')
    lines += ["", "| Paired learner change, Structured − MLP | Low (pp) | Medium (pp) | High (pp) | Equal (pp) |", "|---|---:|---:|---:|---:|"]
    for selection in ("best", "last"):
        ss=result["seed_deltas"][selection]
        lines.append('| '+selection+' | '+' | '.join(f"{100*ss[k]['mean']:+.2f} ± {100*ss[k]['sample_sd_across_seeds']:.2f}" for k in (*LOADS,"mean"))+' |')
    trajectory = [r['rates']['mean']['mean'] for r in audit['pointwise_mean_and_sample_sd']]
    gains = [audit['tail_changes'][str(seed)]['last_interval_gain_pp'] for seed in SEEDS]
    lines += ["", "## Training trajectory", "",
              f"The pointwise equal-load mean changes from {100*trajectory[1]:.2f}% at 102,400 steps to {100*trajectory[-1]:.2f}% at 1,024,000 steps ({100*(trajectory[-1]-trajectory[1]):+.2f} pp). The last three checkpoint means are " + ' / '.join(f'{100*v:.2f}%' for v in trajectory[-3:]) + '.',
              "Final-interval changes for seeds 1–5 (pp): " + ' / '.join(f'{v:+.2f}' for v in gains) + '.',
              "Seeds with best before last: " + str(audit['seeds_best_before_last']) + '. These finite-budget trajectories do not establish convergence.',
              "", "[Full curves](m4_structured_curves.png) · [Post-100k curves](m4_structured_tail_curves.png) · [Matched-day differences](m4_structured_paired.png)"]
    lines += ["", "Each change uses the corresponding learner seed's selected checkpoint. The complete 1,200 best/last × seed × load × day differences, including reward changes and Slack comparisons, are in `m4_structured_paired.json`. Day-level counts are descriptive; the 600 best day pairs are not 600 independent training repetitions.", "", "## Matched-day differences for selected best models", "", "| Load | Mean change (pp) | Win / tie / loss across 200 seed-day pairs | Days all five win | Days all five lose |", "|---|---:|---:|---:|---:|"]
    for load,r in result['paired_summary']['best'].items():
        lines.append(f"| {load} | {100*r['mean_delta']:+.2f} | {r['win_pairs']} / {r['tie_pairs']} / {r['loss_pairs']} | {r['days_all_five_win']} | {r['days_all_five_lose']} |")
    lines += ["", "## Service and training objective", "", "Each cell is a mean per complete demand day, averaged equally across learner seeds. Baseline rows use the identical days.", "", "| Load / model | On time % | Reward | On-time count | Late count | Expired | Flight energy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for load in LOADS:
        for name in ("slack", "mlp_best", "structured_best", "structured_last"):
            r=result['service'][name][load];lines.append(f"| {load} / {name} | {100*r['on_time_rate']:.2f} | {r['reward']:.3f} | {r['on_time']:.3f} | {r['late_deliveries']:.3f} | {r['expired']:.3f} | {r['energy_used']:.2f} |")
    lines += ["", "| Load / order type | MLP best on time % | Structured best on time % | Slack on time % |", "|---|---:|---:|---:|"]
    for load in ('medium', 'high'):
        for order_type in result['service']['structured_best'][load]['type_daily_rates']:
            rates=[result['service'][name][load]['type_daily_rates'][order_type]['on_time'] for name in ('mlp_best','structured_best','slack')]
            lines.append(f"| {load} / {order_type} | {100*rates[0]:.2f} | {100*rates[1]:.2f} | {100*rates[2]:.2f} |")
    lines += ["", "Per-type rates average complete demand days with at least one order of that type; denominators and other service outcomes are in the JSON."]
    lines += ["", "## M4-A follow-up: seed 4 Dev best vs last", "", "The preceding-update windows are in `m4a_diagnostics.md`. This table completes the daily service/action-time comparison; action durations include flight and handling within the macro-action.", "", "| Load / selection | On time % | Reward | Expiry % | PICK min | DELIVER min | CHARGE min | WAIT min |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for load in LOADS:
        for selection in ('best','last'):
            r=result['m4a_seed4_dev_service_and_action_minutes'][selection][load];a=r['action_minutes_per_day']
            lines.append(f"| {load} / {selection} | {100*r['on_time_rate']:.2f} | {r['reward']:.3f} | {100*r['expiry_rate']:.2f} | {a['pick']:.2f} | {a['deliver']:.2f} | {a['charge']:.2f} | {a['wait']:.2f} |")
    lines += ["", "Seed 4's lower final on-time rate accompanies higher reward and lower expiry in Medium/High. The recorded windows show no KL/value explosion. This remains an observed objective/service tradeoff; the logs do not identify a unique causal explanation."]
    lines += ["", "Per-type daily on-time, delivered, late and expiry rates (with empty-type days excluded and denominators recorded) are in the summary JSON. Action time distributions and per-seed best/last changes are retained there and in the audit JSON.", "", "## Behavior on Medium and High", "", "All selected structured best/last models are replayed on all 80 Medium/High days and checked against their saved reward and on-time records. The M4-A replay supplies the same-day MLP and Slack references. Means below are event-weighted; numerators and denominators are retained.", "", "| Load / model | Boxes/drop | Batch drops % | Picked then expired % | Charge arrival battery | Charge target | Hidden orders/state |", "|---|---:|---:|---:|---:|---:|---:|"]
    for load in ("medium", "high"):
        for name in ("slack", "mlp_best", "structured_best", "structured_last"):
            r=result['behavior'][f'{name}_{load}']['event_weighted_means'];lines.append(f"| {load} / {name} | {r['boxes_per_delivery_action']:.3f} | {100*r['batch_delivery_action_fraction']:.2f} | {100*r['expired_after_pick_fraction']:.2f} | {r['charge_arrival_battery']:.2f} | {r['charge_target']:.2f} | {r['hidden_orders']:.3f} |")
    lines += ["", "Compared with MLP best, the structured best models show higher batching and lower expiry after pickup on both loads. Charging occurs at higher arrival battery and with higher target levels, while mean daily flight energy is lower. The joint service/energy results support the narrower observation that this learned scheduling behavior differs in useful ways; charging frequency alone is not a causal explanation."]
    lines += ["", "These policies visit different states. Batching, charging and expiry differences are associations, not proof of the mechanism causing any score change. Pickup slack, payload, demand at charging, waiting with legal service, legal counts and normalized entropy remain in the full aggregates and raw behavior days.", "", "## Compute and reproducibility", "", "| Architecture | Parameters | Masked argmax batch 1 (µs) | Batch 8 (µs) | Batch 256 (µs) |", "|---|---:|---:|---:|---:|"]
    for name,r in result['inference_benchmark']['measurements'].items():
        times=[r['batch'][str(b)]['masked_argmax']['microseconds_per_batch_median'] for b in (1,8,256)]
        lines.append(f"| {name} | {r['parameters']:,} | {times[0]:.2f} | {times[1]:.2f} | {times[2]:.2f} |")
    lines += ["", "Inference uses one CPU thread, the same 256 archived real observations, 50 warmups, seven repetitions of 300 calls, with no experiment learner running. The structured model has fewer parameters but more shared per-order computation and higher inference latency.", "", "| Seed | Complete training days | Training stages (s) | Actual training transitions/s | Run wall (s) | Process CPU user+system (s) |", "|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        r=audit['run_audits'][str(seed)];t=r['timings'];cpu=result['process_times'][str(seed)]
        lines.append(f"| {seed} | {r['completed_training_days']:,} | {t['training_seconds']:.2f} | {1024000/t['training_seconds']:.1f} | {t['wall_seconds']:.2f} | {cpu['user']+cpu['sys']:.2f} |")
    p=result['pilot_timing'];lines += ["", f"M4 formal runs used five independent local CPU processes, one torch thread each. Orchestrator elapsed time was {audit['orchestrator_wall_seconds']:.2f} s; summed process wall time was {audit['timings']['wall_seconds']:.2f} s. M3 was serial, so elapsed time is not a direct compute comparison. The M4 seed-0 pilot used {p['structured_training_seconds']:.2f} s for training stages ({p['structured_transitions_per_second']:.1f} transitions/s), versus the earlier MLP pilot's {p['mlp_training_seconds']:.2f} s ({p['mlp_transitions_per_second']:.1f} transitions/s); these were separate one-thread pilot measurements, not a controlled simultaneous timing trial.", "", f"All 5,000 updates and 55 full-Dev checkpoints passed the saved-source, budget, selection, finite-tensor, legal-action and complete-day audits. There were {audit['coverage']['complete_training_days']:,} complete training episodes and 6,600 PPO Dev episodes. All five initial parameter digests differ. M3 models, logs and selection rules were preserved. Training source commit: `{audit['source_commit']}`. Frozen protocol: `configs/m4_protocol.json`.", "", "## Interpretation", ""]
    eq=result['seed_deltas']['best']['mean'];gap=audit['best_across_seed_stats']['mean']['mean']-audit['baselines']['slack']['mean_on_time_rate']
    best_equal=audit['best_across_seed_stats']['mean']['mean']
    last_equal=audit['last_across_seed_stats']['mean']['mean']
    reward_change=mean(result['service']['structured_last'][load]['reward']-result['service']['structured_best'][load]['reward'] for load in LOADS)
    lines.append(f"All five corresponding learner comparisons improve (+{100*min(eq['values']):.2f} to +{100*max(eq['values']):.2f} pp). All best checkpoints occur before the final step. The final mean is {100*(last_equal-best_equal):.2f} pp below selected best, while equal-load mean daily reward changes by {reward_change:+.3f}. Medium/High expiry is lower at last despite lower on-time rate. This observed tradeoff does not establish convergence or a unique causal explanation for the decline.")
    lines.append(f"The selected structured architecture changes equal-load Dev on-time rate by {100*eq['mean']:+.2f} ± {100*eq['sample_sd_across_seeds']:.2f} pp across the five corresponding learner seeds, and its mean is {100*gap:+.2f} pp relative to Slack. This is evidence for this specific architecture and training protocol on reused Dev; it neither isolates parameter count from structural bias nor establishes held-out generalization.")
    lines += ["", "No network, reward or PPO tuning followed the pilot. No Medium-only, BC, larger-slot or memory-network experiment was mixed into M4. Test remains sealed until candidate methods and the final comparison protocol are frozen.", ""]
    path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'outputs/m4_structured_ppo')
    parser.add_argument('--output',type=Path,default=ROOT/'results/m4_structured')
    args=parser.parse_args();torch.set_num_threads(1)
    audit=analyze(args.run_dir,args.output.parent/'m4_structured_audit',render=False)
    mlp=read(ROOT/'results/m3_mlp_ppo.json');protocol=read(ROOT/'configs/m4_protocol.json')
    assert audit['protocol_sha256']==digest(ROOT/'configs/m4_protocol.json')
    for config_path, expected in protocol['configuration_sha256'].items():
        assert digest(ROOT/config_path)==expected
    for seed in SEEDS:
        initial=load_checkpoint(args.run_dir/f'seed_{seed}/checkpoint_000000000.pt')
        assert initial['architecture']==protocol['architecture']
        assert sum(v.numel() for v in initial['model_state_dict'].values())==65_232
        old_cfg=read(ROOT/f'configs/ppo_m3_seed{seed}.json')
        new_cfg=read(args.run_dir/f'seed_{seed}/config.json');new_cfg.pop('architecture')
        assert new_cfg==old_cfg
    assert digest(ROOT/'results/m3_mlp_ppo.json')==protocol['mlp_reference']['result_sha256']
    assert audit['source_commit']!=mlp['source_commit']
    old_protocol=read(ROOT/'configs/m3_protocol.json')
    changed=[p for p,h in old_protocol['frozen_file_sha256'].items() if protocol['frozen_file_sha256'][p]!=h]
    assert changed==protocol['source_changes_vs_m3']
    pairs=[];selected=defaultdict(list);indices={};states={}
    for selection in ('best','last'):
        for seed in SEEDS:
            new,index=daily_at(args.run_dir/f'seed_{seed}',selection)
            old,old_index=daily_at(ROOT/f'outputs/m3_mixed_mlp_ppo/seed_{seed}',selection)
            indices[(selection,seed)]=index;states[(selection,seed)]=new
            selected['structured_'+selection].extend(index.values());selected['mlp_'+selection].extend(old_index.values())
            for (load,day),r in index.items():
                o=old_index[(load,day)]
                assert r['released']==o['released'] and {k:v['released'] for k,v in r['type_counts'].items()}=={k:v['released'] for k,v in o['type_counts'].items()}
                pairs.append({'selection':selection,'learner_seed':seed,'load':load,'demand_seed':day,
                              'structured_step':new['steps'],'mlp_step':old['steps'],
                              'structured_on_time_rate':r['on_time_rate'],'mlp_on_time_rate':o['on_time_rate'],
                              'delta_structured_mlp':r['on_time_rate']-o['on_time_rate'],
                              'structured_reward':r['reward'],'mlp_reward':o['reward'],
                              'reward_delta':r['reward']-o['reward']})
    canonical=[r for r in records(ROOT/'outputs/ppo_mixed_102400_seed0/evaluation_episodes.jsonl') if r['policy']=='slack']
    slack_index=day_index(canonical);selected['slack']=canonical
    for r in pairs:
        slack=slack_index[(r['load'],r['demand_seed'])]
        r['slack_on_time_rate']=slack['on_time_rate'];r['delta_structured_slack']=r['structured_on_time_rate']-slack['on_time_rate']
    seed_deltas={};paired_summary={}
    for selection in ('best','last'):
        seed_deltas[selection]={};paired_summary[selection]={}
        for load in (*LOADS,'mean'):
            selected_pairs=[r for r in pairs if r['selection']==selection and (r['load']==load or load=='mean')]
            seed_deltas[selection][load]=stats([mean(r['delta_structured_mlp'] for r in selected_pairs if r['learner_seed']==seed) for seed in SEEDS])
            if load!='mean':
                vals=[r['delta_structured_mlp'] for r in selected_pairs]
                day_vals=[[r['delta_structured_mlp'] for r in selected_pairs if r['demand_seed']==day] for day in DEV_SEEDS]
                paired_summary[selection][load]={'mean_delta':mean(vals),'win_pairs':sum(v>1e-12 for v in vals),'tie_pairs':sum(abs(v)<=1e-12 for v in vals),'loss_pairs':sum(v<-1e-12 for v in vals),'days_all_five_win':sum(all(v>1e-12 for v in vs) for vs in day_vals),'days_all_five_lose':sum(all(v<-1e-12 for v in vs) for vs in day_vals)}
    write_json(args.output.parent/(args.output.name+'_paired.json'),{'unit':'selection x learner seed x load x matched Dev day','pairs':pairs})
    slack_behavior_index={(r['episode_load'],r['demand_seed']):r for r in records(ROOT/'outputs/m4a_diagnostics/behavior_days.jsonl') if r['selection']=='slack'}
    behavior_root=args.run_dir/'behavior';behavior_root.mkdir(exist_ok=True)
    if (behavior_root/'days.jsonl').exists():
        behavior_days=list(records(behavior_root/'days.jsonl'));assert len(behavior_days)==800
    else:
        behavior_days=[];env=MedicalUAVEnv()
        for selection in ('best','last'):
            for seed in SEEDS:
                state=states[(selection,seed)];model=build_network(state['architecture']).eval();model.load_state_dict(state['model_state_dict'])
                for load in ('medium','high'):
                    for day in DEV_SEEDS:
                        r=replay(env,load,day,model=model);original=indices[(selection,seed)][(load,day)]
                        assert r['on_time_rate']==original['on_time_rate'] and r['reward']==original['reward']
                        baseline=slack_behavior_index[(load,day)]
                        assert r['order_fingerprint']==baseline['order_fingerprint']
                        r.update(selection=selection,learner_seed=seed);append_json(behavior_root/'days.jsonl',r);behavior_days.append(r)
                print(f'Behavior replay {selection}, seed {seed}',flush=True)
        env.close()
    assert len({(r['selection'],r['learner_seed'],r['episode_load'],r['demand_seed']) for r in behavior_days})==800
    for r in behavior_days:
        original=indices[(r['selection'],r['learner_seed'])][(r['episode_load'],r['demand_seed'])]
        assert r['reward']==original['reward'] and r['on_time_rate']==original['on_time_rate']
        assert r['order_fingerprint']==slack_behavior_index[(r['episode_load'],r['demand_seed'])]['order_fingerprint']
    behavior={}
    for load in ('medium','high'):
        for selection in ('best','last'):
            behavior[f'structured_{selection}_{load}']=aggregate_behavior([r for r in behavior_days if r['selection']==selection and r['episode_load']==load])
    m4a=read(ROOT/'results/m4a_diagnostics.json')
    for key,r in m4a['behavior'].items():
        name,load=key.rsplit('_',1);behavior[(name if name=='slack' else 'mlp_'+name)+'_'+load]=r
    # Seed 4's Dev action-time and service changes accompany its update windows.
    seed4={}
    for selection in ('best','last'):
        _, idx=daily_at(ROOT/'outputs/m3_mixed_mlp_ppo/seed_4',selection)
        seed4[selection]=service_table(list(idx.values()))
    process_times={}
    for seed in SEEDS:
        log=(args.run_dir/f'seed_{seed}.log').read_text()
        process_times[str(seed)]={key:float(re.search(r'^'+key+r' (\d+\.\d+)\s*$',log,re.M).group(1)) for key in ('real','user','sys')}
    p=read(ROOT/'outputs/m4_structured_pilot_seed0/summary.json');oldp=read(ROOT/'outputs/ppo_mixed_102400_seed0/summary.json')
    reload_file=args.run_dir/'reload_verification.json'
    reload_verified=None
    if reload_file.exists():
        reloaded=read(reload_file)['episodes']
        for row in reloaded:
            original=indices[('best',1)][(row['episode_load'],row['demand_seed'])]
            assert row=={k:v for k,v in original.items() if k not in ('transitions','update_index')}
        reload_verified={'episodes':len(reloaded),'all_fields_exact':True,'checkpoint_sha256':digest(args.run_dir/'seed_1/best.pt'),
                         'method':'standalone scripts/evaluate.py, explicit Dev subset'}
    result={'milestone':'M4','source_commit':audit['source_commit'],'standalone_reload_verification':reload_verified,'seed_deltas':seed_deltas,'paired_summary':paired_summary,
            'best_across_seed_stats':audit['best_across_seed_stats'],'last_across_seed_stats':audit['last_across_seed_stats'],
            'service':{name:service_table(group) for name,group in selected.items()},'behavior':behavior,
            'inference_benchmark':read(ROOT/'results/m4_inference_benchmark.json'),'process_times':process_times,
            'pilot_timing':{'structured_training_seconds':p['training_seconds'],'mlp_training_seconds':oldp['training_seconds'],
                            'structured_transitions_per_second':102400/p['training_seconds'],'mlp_transitions_per_second':102400/oldp['training_seconds']},
            'coverage':audit['coverage'],'tail_changes':audit['tail_changes'],'test_episodes':0,
            'm4a_seed4_dev_service_and_action_minutes':seed4,
            'audit_json':'m4_structured_audit.json','behavior_replay_days':800,
            'notes':['Mean +/- sample SD over five learner seeds is not a confidence interval.',
                     'Day-level paired distributions do not substitute for training repetitions.',
                     'Same seed numbers do not imply same trajectories across architectures.',
                     'Different policies visit different states; behavioral differences are descriptive.',
                     'Test remains unused; these are selected and reused Dev comparisons.']}
    write_json(args.output.with_suffix('.json'),result)
    report(result,audit,mlp,args.output.with_suffix('.md'))
    charts(audit,mlp,pairs,args.output)
    print('M4 audited results, report, paired differences and figures saved.',flush=True)


if __name__=='__main__':main()
