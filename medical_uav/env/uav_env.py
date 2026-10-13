"""Gymnasium wrapper around the simulator."""
import copy
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from medical_uav.env.actions import N_ACTIONS, WAIT, action_name, decode
from medical_uav.env.config import NODE_NAMES, ORDER_TYPES, EnvConfig
from medical_uav.env.map_gen import generate_map
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import OBS_BOUND, OBS_DIM, build_observation, rank_pending
from medical_uav.env.orders import generate_orders
from medical_uav.env.simulator import Simulator
from medical_uav.splits import TRAIN_SEED_RANGE


class MedicalUAVEnv(gym.Env):
    """One day of medical pickup and delivery with a single drone.

    Each step runs one action to completion. An action the mask forbids is replaced by WAIT.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, load: str = "medium",
                 render_mode: str | None = None):
        self.cfg = copy.deepcopy(config) if config is not None else EnvConfig()  # private copy
        self.cfg.validate()
        if load not in self.cfg.loads:
            raise ValueError(f"unknown load {load!r}, expected one of {list(self.cfg.loads)}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")
        self.load = load
        self.render_mode = render_mode
        self.city_map = generate_map(self.cfg.map_seed, self.cfg)

        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.lam = self.cfg.loads[load]
        self.demand_seed = None
        self.decision_log: list[tuple[float, str, float]] = []  # (time, action, reward)
        self.sim: Simulator | None = None
        self.slots = []
        self._mask = np.zeros(N_ACTIONS, dtype=bool)
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._done = False
        self._renderer = None

    def reset(self, seed=None, options=None):
        """Start a new day.

        Options:
            demand_seed: seed for the day's orders; a random training seed if omitted.
            load: load level name; defaults to the one given at construction.
            lam: base arrival rate in orders per minute; overrides load.
        """
        super().reset(seed=seed)
        options = options or {}
        load = options.get("load", self.load)
        if load not in self.cfg.loads:
            raise ValueError(f"unknown load {load!r}, expected one of {list(self.cfg.loads)}")
        self.lam = float(options.get("lam", self.cfg.loads[load]))
        if not (math.isfinite(self.lam) and self.lam >= 0):
            raise ValueError(f"lam must be a finite rate >= 0, got {self.lam}")
        self.demand_seed = options.get("demand_seed")
        if self.demand_seed is None:
            self.demand_seed = int(self.np_random.integers(*TRAIN_SEED_RANGE))

        orders = generate_orders(self.demand_seed, self.lam, self.cfg)
        self.sim = Simulator(self.cfg, self.city_map, orders, self.lam)
        self._done = False
        self.decision_log = []
        self._update_view()
        return self._obs.copy(), self._info(0.0, False, "RESET", [])

    def step(self, action):
        """Run one action and return (obs, reward, terminated, truncated, info)."""
        if self.sim is None or self._done:
            raise RuntimeError("episode is over, call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is not in Discrete({N_ACTIONS})")
        action = int(action)
        invalid = not self._mask[action]
        if invalid:
            action = WAIT

        sim = self.sim
        label = self.describe(action)
        t0, r0, n_events = sim.t, sim.total_reward, len(sim.events)
        self._execute(action)
        reward = sim.total_reward - r0
        self.decision_log.append((t0, label, reward))
        self._done = sim.is_done()
        self._update_view()
        info = self._info(sim.t - t0, invalid, action_name(action), sim.events[n_events:])
        return self._obs.copy(), float(reward), self._done, False, info

    def action_masks(self) -> np.ndarray:
        """Return the mask of valid actions."""
        return self._mask.copy()

    def describe(self, action: int) -> str:
        """Return a readable action name; PICK shows the order instead of the slot number."""
        kind, arg = decode(action)
        if kind == "pick" and arg < len(self.slots):
            o = self.slots[arg]
            route = f"{NODE_NAMES[o.origin]}->{NODE_NAMES[o.dest]}"
            return f"PICK {ORDER_TYPES[o.type_idx].name} {route}"
        return action_name(action)

    def render(self):
        if self.render_mode is None or self.sim is None:
            return None
        from medical_uav.env.render import Renderer

        if self._renderer is None:
            self._renderer = Renderer(self.city_map, self.cfg)
        frame = self._renderer.draw(self.sim, self.sim.t)
        if self.render_mode == "rgb_array":
            return frame
        import matplotlib.pyplot as plt

        plt.figure("MedicalUAVEnv")
        plt.clf()
        plt.imshow(frame)
        plt.axis("off")
        plt.pause(1 / self.metadata["render_fps"])
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
        self.slots = rank_pending(self.sim)
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
