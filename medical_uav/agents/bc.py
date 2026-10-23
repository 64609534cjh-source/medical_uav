"""Behaviour cloning: record a teacher's decisions and train a network to imitate them."""
import copy
from multiprocessing import Pool

import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import decode

LOADS = ("low", "medium", "high")  # stored as indices in the "load" array


def _episode(job):
    """Run one day with the teacher; return its (obs, mask, action) records."""
    teacher, load, seed = job
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    records, done = [], False
    while not done:
        a = teacher(env)
        records.append((obs, info["action_mask"], a))
        obs, _, done, _, info = env.step(a)
    return load, seed, records


def collect_demonstrations(teacher, loads: list[str], demand_seeds, processes: int = 1) -> dict:
    """Record the teacher's decision at every step of every (load, demand seed) day.

    Returns arrays obs [N, 452], mask [N, 26], action [N], seed [N] and load [N] (index into LOADS).
    """
    jobs = [(teacher, load, int(seed)) for load in loads for seed in demand_seeds]
    if processes > 1:
        with Pool(processes) as pool:
            episodes = pool.map(_episode, jobs)
    else:
        episodes = [_episode(job) for job in jobs]
    obs, mask, action, seeds, load_idx = [], [], [], [], []
    for load, seed, records in episodes:
        for o, m, a in records:
            obs.append(o)
            mask.append(m)
            action.append(a)
            seeds.append(seed)
            load_idx.append(LOADS.index(load))
    return {"obs": np.array(obs, dtype=np.float32), "mask": np.array(mask, dtype=bool),
            "action": np.array(action, dtype=np.int64), "seed": np.array(seeds, dtype=np.int64),
            "load": np.array(load_idx, dtype=np.int64)}


def split_by_seed(data: dict, val_fraction: float = 0.2, seed: int = 0) -> tuple[dict, dict]:
    """Split by demand seed: all of a seed's decisions (at every load) go to the same side."""
    seeds = np.unique(data["seed"])
    val_seeds = np.random.default_rng(seed).permutation(seeds)[:round(len(seeds) * val_fraction)]
    is_val = np.isin(data["seed"], val_seeds)
    return ({k: v[~is_val] for k, v in data.items()}, {k: v[is_val] for k, v in data.items()})


def action_families(actions: np.ndarray) -> np.ndarray:
    """Return the action kind (pick, deliver, charge, return or wait) of each action index."""
    return np.array([decode(int(a))[0] for a in actions])


def _loss_and_accuracy(net, data: dict, batch_size: int, device: str) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and the per-decision correctness of the greedy action."""
    losses, correct = [], []
    with torch.no_grad():
        for i in range(0, len(data["action"]), batch_size):
            obs, mask, target = (torch.as_tensor(data[k][i:i + batch_size]).to(device)
                                 for k in ("obs", "mask", "action"))
            logits, _ = net(obs, mask)
            losses.append(torch.nn.functional.cross_entropy(logits, target, reduction="sum").item())
            correct.append((logits.argmax(1) == target).cpu().numpy())
    return sum(losses) / len(data["action"]), np.concatenate(correct)


def train_bc(net, train: dict, val: dict, epochs: int = 30, batch_size: int = 256,
             lr: float = 3e-4, seed: int = 0, device: str = "cpu") -> list[dict]:
    """Train the policy head to predict the teacher's action (cross-entropy on masked logits).

    Keeps the weights of the epoch with the lowest validation loss. The value head is not trained.
    Returns one dict per epoch: train_loss, val_loss, val_acc and val_acc_by_family.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tensors = {k: torch.as_tensor(train[k]) for k in ("obs", "mask", "action")}
    families = action_families(val["action"])
    history, best = [], (float("inf"), None)
    for _ in range(epochs):
        net.train()
        order, total = rng.permutation(len(train["action"])), 0.0
        for i in range(0, len(order), batch_size):
            batch = torch.as_tensor(order[i:i + batch_size])
            obs, mask, target = (tensors[k][batch].to(device) for k in ("obs", "mask", "action"))
            logits, _ = net(obs, mask)
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        net.eval()
        val_loss, correct = _loss_and_accuracy(net, val, batch_size, device)
        history.append({"train_loss": total / len(order), "val_loss": val_loss,
                        "val_acc": float(correct.mean()),
                        "val_acc_by_family": {str(f): float(correct[families == f].mean())
                                              for f in np.unique(families)}})
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])
    return history
