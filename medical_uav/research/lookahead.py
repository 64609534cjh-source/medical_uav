"""Bounded planning from public observations and independent synthetic futures.

The online API never receives a live environment, simulator or demand-day seed.
Current state reconstruction is deliberately conservative; missing recent-arrival
history, hidden orders and roundtrip inconsistencies cause exact-policy fallback.
"""
import copy
from collections import Counter
import hashlib
import json
import time

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from medical_uav.agents.destination_inference import DestinationInferencePolicy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import DELIVER0, N_ACTIONS, WAIT
from medical_uav.env.config import EnvConfig, ORDER_TYPES, day_length
from medical_uav.env.map_gen import generate_map
from medical_uav.env.observation import OBS_BOUND, OBS_DIM
from medical_uav.env.orders import Order, generate_orders
from medical_uav.env.simulator import Simulator


class Ineligible(ValueError):
    """Public information is insufficient for the conservative prototype."""


class BudgetExceeded(RuntimeError):
    pass


def hot_index(values):
    index = int(np.argmax(values))
    expected = np.zeros_like(values)
    expected[index] = 1
    if not np.array_equal(values, expected):
        raise Ineligible('invalid_public_one_hot')
    return index


class PublicSnapshot:
    """Reconstruct only fields recoverable from the 403 public features."""
    def __init__(self, observation, mask, config, city_map):
        self.observation = np.array(observation, dtype=np.float32, copy=True)
        self.mask = np.array(mask, dtype=bool, copy=True)
        self.config = copy.deepcopy(config)
        self.city_map = city_map
        x = self.observation
        if x.shape != (403,) or self.mask.shape != (26,) or not np.isfinite(x).all():
            raise Ineligible('invalid_public_shape_or_values')
        if np.abs(x).max() >= OBS_BOUND:
            raise Ineligible('clipped_public_features')
        if x[25] != 0:
            raise Ineligible('hidden_pending_orders')
        self.t = float(x[0])*day_length(config)
        self.location = hot_index(x[9:16])
        self.battery = float(x[6])*100
        self.lam = float(x[5])*config.loads['high']
        self.current = []
        for row in x[31:355].reshape(12, 27):
            if row[0] == 0:
                continue
            kind = hot_index(row[1:5])
            origin, destination = hot_index(row[5:11]), hot_index(row[11:17])
            self._add_order(kind, origin, destination, float(row[17]), float(row[18]), 'pending')
        for row in x[355:].reshape(3, 16):
            if row[0] == 0:
                continue
            kind = hot_index(row[1:5])
            self._add_order(kind, 0, hot_index(row[5:11]), float(row[11]), float(row[12]),
                            'expired' if row[14] else 'onboard')
        # Cargo origin/pickup history are absent from the observation. They do
        # not affect future physical transitions; no historical metrics are used.
        # Recent arrivals DO affect future observations, so do not invent the
        # release times of already completed orders missing from public rows.
        recent = sorted(o.release for o in self.current if o.release > self.t-30)
        if len(recent) != round(float(x[30])*10):
            raise Ineligible('unrepresented_recent_arrival_history')
        self.recent = recent
        reconstructed = self.build_world([])
        if not np.array_equal(reconstructed.action_masks(), self.mask):
            raise Ineligible('public_mask_roundtrip_mismatch')
        if not np.allclose(reconstructed._obs, x, atol=5e-5, rtol=1e-5):
            raise Ineligible('public_observation_roundtrip_mismatch')

    def _add_order(self, kind, origin, destination, weight, deadline_feature, status):
        definition = ORDER_TYPES[kind]
        release = max(0., self.t+deadline_feature*definition.limit-definition.limit)
        if release > self.t+5e-5:
            raise Ineligible('public_release_after_current_time')
        order = Order(len(self.current), kind, origin, destination, weight,
                      min(release, self.t), definition.limit, definition.expiry_after, status=status)
        if status != 'pending':
            order.pickup_time = self.t
        if status == 'expired':
            order.end_time = order.expiry
        self.current.append(order)

    def build_world(self, future):
        """A fresh synthetic world: never a copy of the live simulator."""
        if any(o.release <= self.t or o.status != 'future' for o in future):
            raise ValueError('synthetic future must start after the public clock')
        cfg = copy.deepcopy(self.config)
        orders = copy.deepcopy(self.current + list(future))
        orders.sort(key=lambda o: (o.release, o.id))
        sim = Simulator(cfg, self.city_map, [], self.lam)
        sim.orders = orders
        sim.t, sim.location, sim.battery = self.t, self.location, self.battery
        sim.pending = [o for o in orders if o.status == 'pending']
        sim.onboard = [o for o in orders if o.status in ('onboard', 'expired')]
        sim._next_release = len(self.current)
        sim.release_times_log = list(self.recent)
        # Initialize the frozen wrapper's public stepping fields without reset(),
        # which would generate a demand day. Map geometry is declared static context.
        env = MedicalUAVEnv.__new__(MedicalUAVEnv)
        env.cfg, env.city_map, env.sim = cfg, self.city_map, sim
        env.lam, env.demand_seed = self.lam, None
        env.load = min(cfg.loads, key=lambda k: abs(cfg.loads[k]-self.lam))
        env.render_mode, env._renderer = None, None
        env.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, (OBS_DIM,), np.float32)
        env.action_space = spaces.Discrete(N_ACTIONS)
        env._done, env.decision_log = sim.is_done(), []
        env._update_view()
        return env


class PublicLookaheadPolicy(nn.Module):
    """Soft wall-clock budget, complete candidate comparison or baseline fallback."""
    def __init__(self, model, config=None, scenarios=4, horizon=30., value_tail=True,
                 budget_ms=20., planner_seed=71):
        super().__init__()
        if model.architecture != 'destination_mean_403_26_v1':
            raise ValueError('prototype supports the frozen destination model only')
        if scenarios < 1 or horizon <= 0 or (budget_ms is not None and budget_ms < 0):
            raise ValueError('invalid planning budget')
        self.base = DestinationInferencePolicy(model)
        self.config = copy.deepcopy(config or EnvConfig())
        self.city_map = generate_map(self.config.map_seed, self.config)
        self.scenarios, self.horizon = scenarios, horizon
        self.value_tail, self.budget_ms = value_tail, budget_ms
        self.rng = np.random.default_rng(planner_seed)
        self.counts = Counter()
        self.latencies_ms = []
        self.last_scenario_signatures = []
        self.last_scores = []

    @property
    def model(self):
        return self.base.model

    def forward(self, observations):
        return self.model(observations)

    def _check_budget(self, started):
        if self.budget_ms is not None and (time.perf_counter()-started)*1000 > self.budget_ms:
            raise BudgetExceeded()

    @torch.inference_mode()
    def _score(self, env, action, started):
        deadline = env.sim.t+self.horizon
        steps = 0
        while True:
            _, reward, terminated, truncated, info = env.step(action)
            self.counts['extra_simulator_transitions'] += 1
            steps += 1
            if info['invalid_action'] or not np.isfinite(reward) or truncated:
                raise RuntimeError('invalid synthetic rollout')
            env.sim.check_invariant()
            self._check_budget(started)
            if terminated or env.sim.t >= deadline:
                break
            if steps >= 256:
                raise RuntimeError('synthetic rollout exceeded decision cap')
            obs = torch.from_numpy(env._obs.copy())
            mask = torch.from_numpy(env.action_masks())
            action = int(self.base.act(obs, mask, True)[0])
        score = env.sim.total_reward
        # Complete a macro action that crosses H; bootstrap at its actual end.
        if self.value_tail and not terminated:
            score += float(self.model(torch.from_numpy(env._obs.copy()))[1])
        self._check_budget(started)
        return score

    @torch.inference_mode()
    def act(self, observation, mask, deterministic=False):
        if not deterministic or observation.shape != (403,) or observation.device.type != 'cpu':
            raise ValueError('CPU deterministic single-state planning only')
        started = time.perf_counter()
        action = int(self.base.act(observation, mask, True)[0])
        original = action
        self.last_original_action = original
        self.last_outcome = 'baseline'
        self.counts['decisions'] += 1
        self.last_scenario_signatures, self.last_scores = [], []
        try:
            self._check_budget(started)
            if not observation[31:355].reshape(12, 27)[:, 0].any() and not observation[355:].reshape(3, 16)[:, 0].any():
                raise Ineligible('no_released_work')
            snapshot = PublicSnapshot(observation.numpy(), mask.numpy(), self.config, self.city_map)
            self.counts['eligible_states'] += 1
            logits = self.model(observation)[0]
            legal = torch.where(mask)[0].tolist()
            top = sorted(legal, key=lambda k: (-float(logits[k]), k))[:3]
            candidates = list(dict.fromkeys([original]+top))
            live = [o for o in snapshot.current if o.status == 'onboard']
            if live:
                urgent = DELIVER0+min(live, key=lambda o: (o.deadline, o.id)).dest
                if mask[urgent] and urgent not in candidates:
                    candidates.append(urgent)
            if WAIT not in candidates:
                candidates.append(WAIT)
            candidates = candidates[:6]
            worlds = []
            for _ in range(self.scenarios):
                seed = int(self.rng.integers(0, 2**63-1))
                future = [o for o in generate_orders(seed, snapshot.lam, self.config) if o.release > snapshot.t]
                for i, order in enumerate(future):
                    order.id = len(snapshot.current)+i
                signature = [(o.release, o.type_idx, o.origin, o.dest, o.weight) for o in future]
                self.last_scenario_signatures.append(hashlib.sha256(json.dumps(signature).encode()).hexdigest())
                worlds.append(snapshot.build_world(future))
                self.counts['synthetic_scenarios'] += 1
                self._check_budget(started)
            scores = []
            for candidate in candidates:
                # These worlds contain only reconstructed current state and
                # independent futures. Each candidate gets the same absolute-time
                # scenario, irrespective of how many RNG draws its path would take.
                value = np.mean([self._score(copy.deepcopy(world), candidate, started) for world in worlds])
                scores.append(float(value))
            self.last_scores = list(zip(candidates, scores))
            best = int(np.argmax(scores))
            if scores[best] > scores[0]+1e-8:
                action = candidates[best]
            self.counts['completed_comparisons'] += 1
            self.counts['changed_actions'] += action != original
            self.last_outcome = 'completed'
        except Ineligible as error:
            self.counts['fallback_'+str(error)] += 1
            self.last_outcome = str(error)
        except BudgetExceeded:
            self.counts['fallback_budget'] += 1
            self.last_outcome = 'budget'
            action = original
        if not mask[action]:
            raise RuntimeError('planner emitted illegal live action')
        self.latencies_ms.append((time.perf_counter()-started)*1000)
        return torch.tensor(action, dtype=torch.long), None, None

    def planning_summary(self):
        return {'counts': dict(self.counts),
                'latency_ms_p50_p95_max': ([float(np.quantile(self.latencies_ms, q)) for q in (.5, .95)]
                                         + [max(self.latencies_ms)]) if self.latencies_ms else None,
                'budget_ms': self.budget_ms,
                'scope': 'soft budget checked after macro simulation steps; overshoot is measured; '
                         'no hard real-time guarantee; independent synthetic futures, no real demand seed'}
