"""M2 demo: create the two networks, run a forward and a backward pass on real environment data,
then let the trained networks run a whole day.

Usage:
    python scripts/demo_network.py           # about 15 s on a CPU
    python scripts/demo_network.py --quick   # smaller batch, low-load day
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.networks import MLPNet, SetNet, count_parameters
from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import N_SLOTS
from medical_uav.evaluation import run_day
from medical_uav.metrics import summary_lines

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO / "checkpoints"


def record_states(load: str, seed: int, n: int):
    """Run Slack-aware for one day and keep the first n decisions: observations, masks, the
    teacher's actions, readable action names, and the reward still to come at each decision."""
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    observations, masks, actions, names, rewards, done = [], [], [], [], [], False
    while not done:
        a = slack_aware_policy(env)
        observations.append(obs)
        masks.append(info["action_mask"])
        actions.append(a)
        names.append([env.describe(k) for k in range(len(info["action_mask"]))])
        obs, r, done, _, info = env.step(a)
        rewards.append(r)
    to_go = np.cumsum(rewards[::-1])[::-1]  # reward from each decision to the end of the day
    return (torch.as_tensor(np.array(observations[:n])), torch.as_tensor(np.array(masks[:n])),
            torch.as_tensor(actions[:n]), names[:n],
            torch.as_tensor(to_go[:n].copy(), dtype=torch.float32))


def grad_norms(net: torch.nn.Module) -> str:
    """Gradient norm of each top-level part of the network."""
    parts = []
    for name, module in net.named_children():
        grads = [p.grad.norm() for p in module.parameters() if p.grad is not None]
        if grads:
            parts.append(f"{name} {torch.stack(grads).norm():.3f}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    load, batch = ("low", 8) if args.quick else ("medium", 64)
    torch.manual_seed(0)

    # 1. create the networks
    print("1. Networks")
    for net in (MLPNet(), SetNet()):
        print(f"   {type(net).__name__:6s} {count_parameters(net):>7,d} parameters")

    # 2. a batch of real observations and masks
    obs, mask, teacher, names, to_go = record_states(load, 100_000, batch)
    print(f"\n2. Input: observations {tuple(obs.shape)}, action masks {tuple(mask.shape)} "
          f"({load} load, demand seed 100000, first {batch} decisions of Slack-aware)")

    # 3. forward pass with the trained set network
    trained, info = load_checkpoint(CHECKPOINTS / "bc_set.pt")
    with torch.no_grad():
        logits, value = trained(obs, mask)
    print(f"\n3. forward pass (trained set network): logits {tuple(logits.shape)}, "
          f"value {tuple(value.shape)}")
    busiest = torch.argsort(mask.sum(1), descending=True)[:3]  # states with the most valid actions
    for i in sorted(int(k) for k in busiest):
        p = torch.softmax(logits[i], 0)
        top = torch.topk(p, 3)
        choices = ", ".join(f"{names[i][a]} {q:.2f}"
                            for q, a in zip(top.values, top.indices) if q > 0)
        print(f"   decision {i + 1}: {choices}   (Slack-aware chose {names[i][int(teacher[i])]})")
        valid = torch.nonzero(mask[i])[:, 0]
        valid = valid[torch.argsort(logits[i, valid], descending=True)][:3]
        print("      their logits: "
              + ", ".join(f"{names[i][a]} {logits[i, a]:.2f}" for a in valid)
              + f";  value estimate {value[i]:.3f} (untrained: BC does not train the value head)")

    # 4. backward pass: imitation loss for the policy, return-to-go loss for the value head
    net = SetNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    logits, value = net(obs, mask)
    policy_loss = torch.nn.functional.cross_entropy(logits, teacher)
    value_loss = ((value - to_go / 10) ** 2).mean()  # the learner uses rewards / 10
    loss = policy_loss + 0.5 * value_loss
    opt.zero_grad()
    loss.backward()
    print(f"\n4. backward pass (untrained set network): policy loss {policy_loss.item():.3f}, "
          f"value loss {value_loss.item():.3f}")
    print(f"   gradient norms: {grad_norms(net)}")
    opt.step()
    with torch.no_grad():
        logits, value = net(obs, mask)
        after = (torch.nn.functional.cross_entropy(logits, teacher)
                 + 0.5 * ((value - to_go / 10) ** 2).mean())
    print(f"   after one Adam step the loss falls from {loss.item():.3f} to {after.item():.3f}")

    # 5. the trained networks run a whole day on their own
    print(f"\n5. A whole day ({load} load, demand seed 100000) driven by each trained network")
    for kind in ("mlp", "set"):
        net, info = load_checkpoint(CHECKPOINTS / f"bc_{kind}.pt")
        m = run_day(NetworkPolicy(net), load, 100_000)
        print(f"   BC {kind} network (validation accuracy {info['val_acc']:.3f}): "
              f"{100 * m['on_time_rate']:.1f}% on time")
        m["on_time_by_type"] = {k[8:]: v for k, v in m.items()
                                if k.startswith("on_time_") and k != "on_time_rate"}
        for line in summary_lines(m):
            print("     " + line)
    m = run_day(lambda env, o, msk: slack_aware_policy(env), load, 100_000)
    print(f"   Slack-aware (the teacher) on the same day: {100 * m['on_time_rate']:.1f}% on time")
    print(f"\n   (the first {N_SLOTS} logits are PICK actions, one per pending-order slot)")


if __name__ == "__main__":
    main()
