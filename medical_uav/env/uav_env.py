"""Gymnasium wrapper around the simulator (§5, §6, §7)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from medical_uav.env.actions import N_ACTIONS, WAIT, action_name, decode
from medical_uav.env.config import EnvConfig
from medical_uav.env.map_gen import generate_map
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import OBS_BOUND, OBS_DIM, build_observation, rank_pending
from medical_uav.env.orders import generate_orders
from medical_uav.env.simulator import Simulator


class MedicalUAVEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, load: str = "medium",
                 render_mode: str | None = None):
        self.cfg = config or EnvConfig()
        self.load = load
        self.render_mode = render_mode
        self.city_map = generate_map(self.cfg.map_seed, self.cfg)

        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.lam = self.cfg.loads[load]
        self.demand_seed = None
        self.sim: Simulator | None = None
        self.slots = []
        self._mask = np.zeros(N_ACTIONS, dtype=bool)
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._done = False
        self._renderer = None

    def reset(self, seed=None, options=None):
        """options: demand_seed (int), load ("low"/"medium"/"high") or lam (float)."""
        super().reset(seed=seed)
        options = options or {}
        load = options.get("load", self.load)
        self.lam = options.get("lam", self.cfg.loads[load])
        self.demand_seed = options.get("demand_seed", int(self.np_random.integers(2**31 - 1)))

        orders = generate_orders(self.demand_seed, self.lam, self.cfg)
        self.sim = Simulator(self.cfg, self.city_map, orders, self.lam)
        self._done = False
        self._update_view()
        return self._obs.copy(), self._info(0.0, False, "RESET", [])

    def step(self, action):
        if self.sim is None or self._done:
            raise RuntimeError("episode is over, call reset()")
        action = int(action)
        invalid = not self._mask[action]
        if invalid:
            action = WAIT

        sim = self.sim
        t0, r0, n_events = sim.t, sim.total_reward, len(sim.events)
        self._execute(action)
        reward = sim.total_reward - r0
        self._done = sim.is_done()
        self._update_view()
        info = self._info(sim.t - t0, invalid, action_name(action), sim.events[n_events:])
        return self._obs.copy(), float(reward), self._done, False, info

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def render(self):
        return None

    def _execute(self, action: int) -> None:
        kind, arg = decode(action)
        if kind == "pick":
            self.sim.do_pick(self.slots[arg])
        elif kind == "deliver":
            self.sim.do_deliver(arg)
        elif kind == "charge":
            self.sim.do_charge(*arg)
        elif kind == "return":
            self.sim.do_return()
        else:
            self.sim.do_wait()

    def _update_view(self) -> None:
        self.slots = rank_pending(self.sim, self.cfg.n_slots)
        self._mask = compute_action_mask(self.sim, self.slots)
        self._obs = build_observation(self.sim, self.slots, self._mask)

    def _info(self, elapsed, invalid, name, events) -> dict:
        return {
            "action_mask": self._mask.copy(),
            "t": self.sim.t,
            "elapsed": elapsed,
            "invalid_action": invalid,
            "action_name": name,
            "events": list(events),
        }
