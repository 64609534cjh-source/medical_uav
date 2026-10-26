"""Batch neural inference with serial environment steps and explicit resets.

No autoreset wrappers: the post-step observation must still belong to the old day.
Only this module draws training days/loads; it never touches Dev or Test seeds.
"""

from collections import Counter

import numpy as np
import torch

from medical_uav.algorithms.rollout_buffer import RolloutBuffer
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import decode
from medical_uav.metrics import episode_record
from medical_uav.splits import TRAIN_SEED_RANGE


class Collector:
    def __init__(self, num_envs=8, seed=0, loads=("medium",), train_seeds=None,
                 env_factory=MedicalUAVEnv, state=None):
        if num_envs < 1 or not loads or len(set(loads)) != len(loads):
            raise ValueError("positive environment count and unique nonempty loads required")
        if set(loads) - {"low", "medium", "high"}:
            raise ValueError("unknown training load")
        if train_seeds is not None and (not train_seeds or any(
                not TRAIN_SEED_RANGE[0] <= s < TRAIN_SEED_RANGE[1] for s in train_seeds)):
            raise ValueError("debug day list must contain only Train seeds")
        self.seed, self.loads, self.train_seeds = seed, tuple(loads), train_seeds
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.transitions_by_load = Counter()
        self.episodes_by_load = Counter()
        self.episode_serial = 0
        if state:
            self.rng.bit_generator.state = state["sampler_rng"]
            self.transitions_by_load.update(state["transitions_by_load"])
            self.episodes_by_load.update(state["episodes_by_load"])
            self.episode_serial = state["episode_serial"]
        self.envs = [env_factory() for _ in range(num_envs)]
        self.obs, self.masks, self.episode_loads = [None] * num_envs, [None] * num_envs, [None] * num_envs
        self.counts, self.minutes = [Counter() for _ in self.envs], [Counter() for _ in self.envs]
        self.decisions = [0] * num_envs
        for i in range(num_envs):
            self._reset(i)

    def _reset(self, i):
        load = self.loads[int(self.rng.integers(len(self.loads)))]
        day = (int(self.rng.choice(self.train_seeds)) if self.train_seeds is not None
               else int(self.rng.integers(*TRAIN_SEED_RANGE)))
        obs, info = self.envs[i].reset(options={"demand_seed": day, "load": load})
        self.obs[i], self.masks[i], self.episode_loads[i] = obs.copy(), info["action_mask"].copy(), load
        self.counts[i], self.minutes[i], self.decisions[i] = Counter(), Counter(), 0
        self.episode_serial += 1

    @torch.no_grad()
    def collect(self, network, steps):
        buffer = RolloutBuffer(steps, len(self.envs))
        records = []
        for t in range(steps):
            observations = torch.from_numpy(np.stack(self.obs))
            masks = torch.from_numpy(np.stack(self.masks))
            actions, log_probs, values = network.act(observations, masks)
            buffer.obs[t].copy_(observations)
            buffer.action_mask[t].copy_(masks)
            buffer.action[t].copy_(actions)
            buffer.old_log_prob[t].copy_(log_probs)
            buffer.old_value[t].copy_(values)
            post_observations, infos, boundaries = [], [], []
            for i, env in enumerate(self.envs):
                action = int(actions[i])
                obs, reward, terminated, truncated, info = env.step(action)
                if info["invalid_action"] or not masks[i, action]:
                    raise RuntimeError(f"illegal action in env {i}, demand {env.unwrapped.demand_seed}")
                base = env.unwrapped
                base.sim.check_invariant()
                if (base.sim.load_kg() > base.cfg.max_weight + 1e-6
                        or len(base.sim.onboard) > base.cfg.max_boxes):
                    raise RuntimeError("cargo capacity violated")
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("non-finite environment transition")
                buffer.reward[t, i] = reward
                buffer.elapsed[t, i] = info["elapsed"]
                buffer.terminated[t, i], buffer.truncated[t, i] = terminated, truncated
                kind = decode(action)[0]
                self.counts[i][kind] += 1
                self.minutes[i][kind] += float(info["elapsed"])
                self.decisions[i] += 1
                self.transitions_by_load[self.episode_loads[i]] += 1
                post_observations.append(obs.copy())
                infos.append(info)
                boundaries.append((terminated, truncated))
            # Crucially, this inference happens before resetting ANY environment.
            _, next_values = network(torch.from_numpy(np.stack(post_observations)))
            buffer.next_value[t].copy_(next_values)
            for i, (terminated, truncated) in enumerate(boundaries):
                if terminated or truncated:
                    record = episode_record(
                        self.envs[i], self.episode_loads[i], self.decisions[i], self.counts[i],
                        self.minutes[i], self.seed, terminated=terminated, truncated=truncated)
                    record.update({"env_index": i, "rollout_step": t})
                    records.append(record)
                    self.episodes_by_load[self.episode_loads[i]] += 1
                    self._reset(i)
                else:
                    self.obs[i], self.masks[i] = post_observations[i], infos[i]["action_mask"].copy()
        return buffer, records

    def state_dict(self):
        """Sampler/counters only; a resumed run explicitly starts NEW training days."""
        return {"sampler_rng": self.rng.bit_generator.state,
                "transitions_by_load": dict(self.transitions_by_load),
                "episodes_by_load": dict(self.episodes_by_load),
                "episode_serial": self.episode_serial,
                "partial_days_discarded_on_resume": [
                    {"env_index": i, "decisions": self.decisions[i],
                     "demand_seed": int(env.unwrapped.demand_seed),
                     "episode_load": self.episode_loads[i]}
                    for i, env in enumerate(self.envs) if self.decisions[i] > 0]}

    def close(self):
        for env in self.envs:
            env.close()
