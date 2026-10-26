"""M4-A: audit Random streams and describe policy behavior without retraining.

Behavior comparisons describe different policy-visited states on matched Dev days;
they are not counterfactual claims that one action is wrong in the same state.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from medical_uav.agents.networks import MLPActorCritic, masked_distribution
from medical_uav.baselines.heuristics import random_policy, slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import CHARGE0, decode
from medical_uav.env.observation import direct_completion_time
from medical_uav.metrics import episode_record
from medical_uav.splits import DEV_SEEDS
from medical_uav.training.io import append_json, write_json
from medical_uav.training.trainer import load_checkpoint

LOADS = ("low", "medium", "high")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def fingerprint(env):
    immutable = [[o.id, o.type_idx, o.origin, o.dest, o.weight, o.release, o.limit, o.expiry_after]
                 for o in env.sim.orders]
    return hashlib.sha256(json.dumps(immutable).encode()).hexdigest()


@torch.no_grad()
def replay(env, load, day, model=None, random_rng=None):
    obs, info = env.reset(options={"demand_seed": day, "load": load})
    order_hash = fingerprint(env)
    totals, counts, minutes = Counter(), Counter(), Counter()
    pickup_ids = []
    sim = env.sim
    original_fly = sim.fly_to
    def observe_flight(node):
        if node != sim.location:
            totals["flights"] += 1
            totals["departure_kg_sum"] += sim.load_kg()
            totals["departure_boxes_sum"] += len(sim.onboard)
            totals["empty_flights"] += int(not sim.onboard)
        return original_fly(node)
    sim.fly_to = observe_flight
    while True:
        mask = info["action_mask"]
        legal = int(mask.sum())
        totals["decisions"] += 1
        totals["legal_count_sum"] += legal
        totals["single_legal_states"] += int(legal == 1)
        totals["pending_sum"] += len(sim.pending)
        totals["hidden_sum"] += len(sim.pending) - len(env.slots)
        totals["hidden_present_states"] += int(len(sim.pending) > len(env.slots))
        if model is not None:
            logits, _ = model(torch.from_numpy(obs))
            distribution = masked_distribution(logits, torch.from_numpy(mask))
            entropy = float(distribution.entropy())
            totals["entropy_sum"] += entropy
            totals["normalized_entropy_sum"] += entropy / math.log(legal) if legal > 1 else 0
            action = int(distribution.probs.argmax())
        else:
            action = random_policy(env, random_rng) if random_rng is not None else slack_aware_policy(env)
        assert mask[action]
        kind, arg = decode(action)
        has_service = bool(mask[:CHARGE0].any())
        if kind == "pick":
            order = env.slots[arg]
            slack = order.deadline - sim.t - direct_completion_time(sim, order)
            totals["picks"] += 1
            totals["pickup_slack_sum"] += slack
            totals["negative_slack_picks"] += int(slack < -1e-9)
            totals["matching_destination_picks"] += int(order.dest in {o.dest for o in sim.onboard if o.status == "onboard"})
            pickup_ids.append(order.id)
        if kind == "charge":
            station, target = arg
            totals["charge_actions"] += 1
            totals["charge_start_battery_sum"] += sim.battery
            totals["charge_arrival_battery_sum"] += sim.battery - sim.energy(sim.location, station, sim.load_kg())
            totals["charge_target_sum"] += target
            totals[f"charge_target_{int(target)}"] += 1
            totals["charge_pending_sum"] += len(sim.pending)
            totals["charge_with_legal_service"] += int(has_service)
        if kind == "wait":
            totals["wait_actions"] += 1
            totals["wait_with_legal_service"] += int(has_service)
        prior_events = len(sim.events)
        obs, _, terminated, truncated, info = env.step(action)
        assert not truncated and not info["invalid_action"]
        if kind == "deliver":
            boxes = sum(event[1] == "deliver" for event in sim.events[prior_events:])
            totals["drop_calls"] += 1
            totals["delivered_at_drops"] += boxes
            totals["batch_drop_calls"] += int(boxes >= 2)
        counts[kind] += 1
        minutes[kind] += info["elapsed"]
        if terminated:
            break
    totals["expired_after_pick"] = sum(o.id in pickup_ids and o.status == "expired" for o in sim.orders)
    result = episode_record(env, load, totals["decisions"], counts, minutes)
    result.update({"order_fingerprint": order_hash, "behavior_totals": dict(totals)})
    return result


def aggregate_behavior(records):
    sums = Counter()
    for record in records:
        sums.update(record["behavior_totals"])
    ratios = {
        "departure_kg": ("departure_kg_sum", "flights"), "departure_boxes": ("departure_boxes_sum", "flights"),
        "empty_flight_fraction": ("empty_flights", "flights"), "boxes_per_delivery_action": ("delivered_at_drops", "drop_calls"),
        "batch_delivery_action_fraction": ("batch_drop_calls", "drop_calls"),
        "pickup_slack_minutes": ("pickup_slack_sum", "picks"), "negative_slack_pick_fraction": ("negative_slack_picks", "picks"),
        "matching_destination_pick_fraction": ("matching_destination_picks", "picks"),
        "expired_after_pick_fraction": ("expired_after_pick", "picks"),
        "charge_start_battery": ("charge_start_battery_sum", "charge_actions"),
        "charge_arrival_battery": ("charge_arrival_battery_sum", "charge_actions"),
        "charge_target": ("charge_target_sum", "charge_actions"),
        "pending_at_charge": ("charge_pending_sum", "charge_actions"),
        "charge_with_legal_service_fraction": ("charge_with_legal_service", "charge_actions"),
        "wait_with_legal_service_fraction": ("wait_with_legal_service", "wait_actions"),
        "legal_actions": ("legal_count_sum", "decisions"), "single_legal_state_fraction": ("single_legal_states", "decisions"),
        "pending_orders": ("pending_sum", "decisions"), "hidden_orders": ("hidden_sum", "decisions"),
        "hidden_present_fraction": ("hidden_present_states", "decisions"),
        "entropy": ("entropy_sum", "decisions"), "entropy_over_log_legal_count": ("normalized_entropy_sum", "decisions"),
    }
    return {"episodes": len(records), "event_totals": dict(sums),
            "event_weighted_means": {key: sums[a] / sums[b] if sums[b] and a in sums else None for key, (a, b) in ratios.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    m3 = root / "outputs/m3_mixed_mlp_ppo"
    summary = json.loads((root / "results/m3_mlp_ppo.json").read_text())
    canonical = rows(root / "outputs/ppo_mixed_102400_seed0/evaluation_episodes.jsonl")
    slack = [r for r in canonical if r["policy"] == "slack"]
    quality, windows = {}, {}
    fields = ("on_time_rate", "reward", "on_time", "late_deliveries", "expired", "energy_used", "served_rate", "expiry_rate", "charge_minutes", "wait_minutes")
    quality["slack"] = {load: {k: mean(r[k] for r in slack if r["episode_load"] == load) for k in fields} for load in LOADS}
    for selection in ("best", "last"):
        collected = []
        for seed in range(1, 6):
            run = m3 / f"seed_{seed}"
            step = load_checkpoint(run / f"{selection}.pt")["steps"]
            daily = [r for r in rows(run / "evaluation_episodes.jsonl") if r["policy"] == "ppo" and r["transitions"] == step]
            collected += daily
            index = step // 1024
            subset = [r for r in rows(run / "updates.jsonl") if index - 25 < r["update_index"] <= index]
            keys = ("learning_rate", "approx_kl", "clip_fraction", "epochs_completed", "optimizer_steps", "value_loss", "explained_variance", "return_mean", "return_std", "advantage_std", "entropy")
            windows[f"seed_{seed}_{selection}"] = {"checkpoint_step": step, "window_updates": [index - 24, index],
                                                       **{k: {"mean": mean(r[k] for r in subset), "min": min(r[k] for r in subset), "max": max(r[k] for r in subset)} for k in keys}}
        quality[selection] = {load: {k: mean(r[k] for r in collected if r["episode_load"] == load) for k in fields} for load in LOADS}
    env = MedicalUAVEnv()
    random_rows, hashes = [], {}
    for protocol in ("legacy", "current"):
        for load in LOADS:
            for day in DEV_SEEDS:
                rng = np.random.default_rng(day) if protocol == "legacy" else np.random.default_rng(np.random.SeedSequence([0, day, LOADS.index(load), 0]))
                record = replay(env, load, day, random_rng=rng)
                key = (load, day)
                if protocol == "legacy":
                    hashes[key] = record["order_fingerprint"]
                else:
                    assert hashes[key] == record["order_fingerprint"]
                    old = next(r for r in canonical if r["policy"] == "random" and r["episode_load"] == load and r["demand_seed"] == day)
                    assert record["on_time_rate"] == old["on_time_rate"] and record["reward"] == old["reward"]
                record["random_protocol"] = protocol
                random_rows.append(record)
                append_json(args.output_dir / "random_days.jsonl", record)
    random_summary = {protocol: {load: mean(r["on_time_rate"] for r in random_rows if r["random_protocol"] == protocol and r["episode_load"] == load) for load in LOADS} for protocol in ("legacy", "current")}
    print("Random audit", random_summary, flush=True)
    behavior = defaultdict(list)
    for selection in ("slack", "best", "last"):
        seeds = (0,) if selection == "slack" else range(1, 6)
        for seed in seeds:
            model = None
            source_rows = slack
            if seed:
                state = load_checkpoint(m3 / f"seed_{seed}/{selection}.pt")
                model = MLPActorCritic()
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                source_rows = [r for r in rows(m3 / f"seed_{seed}/evaluation_episodes.jsonl")
                               if r["policy"] == "ppo" and r["transitions"] == state["steps"]]
            source_index = {(r["episode_load"], r["demand_seed"]): r for r in source_rows}
            for load in ("medium", "high"):
                for day in DEV_SEEDS:
                    record = replay(env, load, day, model=model)
                    source = source_index[(load, day)]
                    assert record["on_time_rate"] == source["on_time_rate"] and record["reward"] == source["reward"]
                    record.update({"selection": selection, "learner_seed": seed})
                    behavior[f"{selection}_{load}"].append(record)
                    append_json(args.output_dir / "behavior_days.jsonl", record)
            print(f"Replayed {selection} seed {seed}", flush=True)
    env.close()
    result = {"quality": quality, "training_windows": windows, "random_protocols": random_summary,
              "random_orders_identical": True, "random_current_matches_m3": True,
              "behavior": {key: aggregate_behavior(values) for key, values in behavior.items()},
              "notes": ["All metrics are Dev-only. No retraining was performed.",
                        "Legacy Random uses default_rng(demand_seed); current uses SeedSequence([0,day,load_index,map_seed]).",
                        "Both draw uniformly from the current legal action list. Order fingerprints agree for every matched day.",
                        "Historical M3 training legal-action counts were not logged; replay entropy and legal counts describe Dev states, not the missing training states.",
                        "Behavior means are event-weighted, with denominators preserved. Policies visit different states; differences do not prove action-level mistakes."]}
    write_json(args.output_dir / "diagnostics.json", result)
    write_json(root / "results/m4a_diagnostics.json", result)


if __name__ == "__main__":
    main()
