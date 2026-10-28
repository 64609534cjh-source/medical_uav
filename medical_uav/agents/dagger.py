"""DAgger records student execution and labels copies of the same visible state."""
from multiprocessing import get_context

import numpy as np
import torch

from medical_uav.agents.bc import LOADS
from medical_uav.baselines.horizon import RollingHorizon
from medical_uav.baselines.scenario import planning_root
from medical_uav.env import MedicalUAVEnv
from medical_uav.splits import DAGGER_SEEDS, TRAINING_SEEDS


def day_schedule(seed: int, iterations: int, days: int, smoke: bool = False):
    """Disjoint per-student medium blocks and fixed low/high replay days."""
    if seed not in TRAINING_SEEDS or iterations < 1 or days < 1:
        raise ValueError("Use a registered training seed and positive iteration/day counts")
    if iterations * days > 500 or days > 400:
        raise ValueError("Schedule exceeds the reserved per-student day block")
    if smoke:
        if iterations != 1 or days != 1:
            raise ValueError("Smoke requires one iteration and one day per load")
        return [[709905]], [709950]
    base = DAGGER_SEEDS.start + 1000 * TRAINING_SEEDS.index(seed)
    medium = [list(range(base + i * days, base + (i + 1) * days)) for i in range(iterations)]
    return medium, list(range(base + 500, base + 500 + days))


def _label(job):
    teacher, root = job
    return int(teacher(root))


def _worker_init():
    torch.set_num_threads(1)


def collect_student_data(policy, demand_seeds, loads=("medium",), processes: int = 1,
                         config=None, teacher=None, label_with_teacher: bool = True) -> dict:
    """Act only with the student; label saved planning roots after collecting its trajectories."""
    if label_with_teacher and tuple(loads) != ("medium",):
        raise ValueError("Teacher queries are restricted to medium load")
    if processes < 1 or processes > 24:
        raise ValueError("Use 1..24 label processes")
    teacher = teacher if teacher is not None else RollingHorizon(depth=3, foresight=False)
    records, roots = [], []
    for load in loads:
        env = MedicalUAVEnv(config=config, load=load)
        for seed in demand_seeds:
            obs, info = env.reset(options={"demand_seed": int(seed)})
            done = False
            while not done:
                mask = info["action_mask"]
                action = int(policy(env, obs, mask))
                if not mask[action]:
                    raise ValueError("Student executed an invalid action")
                if label_with_teacher:
                    root = planning_root(env)
                    if not np.array_equal(root.action_masks(), mask):
                        raise ValueError("Planning root differs from the student's mask")
                    roots.append(root)
                records.append((obs.copy(), mask.copy(), action, int(seed), LOADS.index(load)))
                obs, _, done, truncated, info = env.step(action)
                if truncated:
                    raise ValueError("Incomplete student episode")
    if not records:
        raise ValueError("No student decisions collected")
    obs, mask, action, seeds, load = zip(*records)
    data = {"obs": np.asarray(obs, dtype=np.float32), "mask": np.asarray(mask, dtype=bool),
            "action": np.asarray(action, dtype=np.int64), "seed": np.asarray(seeds, dtype=np.int64),
            "load": np.asarray(load, dtype=np.int64)}
    if label_with_teacher:
        jobs = [(teacher, root) for root in roots]
        if processes > 1:
            with get_context("spawn").Pool(processes, initializer=_worker_init) as pool:
                labels = pool.map(_label, jobs, chunksize=1)
        else:
            labels = list(map(_label, jobs))
        data["label"] = np.asarray(labels, dtype=np.int64)
    else:
        data["label"] = data["action"].copy()
    labels = data["label"]
    if (np.any(labels < 0) or np.any(labels >= data["mask"].shape[1])
            or not data["mask"][np.arange(len(labels)), labels].all()):
        raise ValueError("Teacher label is forbidden by the student's mask")
    return data


def aggregate_datasets(*datasets: dict) -> dict:
    """Retain all earlier teacher states and the fixed replay, in collection order."""
    if not datasets or any(d.keys() != datasets[0].keys() for d in datasets):
        raise ValueError("Dataset schemas differ")
    return {key: np.concatenate([d[key] for d in datasets]) for key in datasets[0]}
