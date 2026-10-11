# M1 — Medical UAV Gym Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, visualise and calibrate the Gymnasium environment for single-UAV dynamic medical pickup-and-delivery, ready for the 10/16 "OpenAI Gym Code and Report" submission.

**Architecture:** A pure-Python event-driven `Simulator` (no Gym dependency) owns time, battery, cargo and orders and executes high-level commands (pick / deliver / charge / return / wait). A thin `MedicalUAVEnv` (Gymnasium) translates the 26 discrete actions into simulator commands, computes the action mask and the 415-dim observation, and delegates rendering to a matplotlib `Renderer` that replays the simulator's recorded history. Maps (grid + no-fly zones + A*) and order streams are generated from seeds before the episode starts, so every policy sees identical inputs.

**Tech Stack:** Python 3.11 (conda env `medical-uav`), numpy, gymnasium 1.x, matplotlib, imageio + imageio-ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-medical-uav-design.md`. Section numbers (§) below refer to it. Executors read both documents.

## Global Constraints

- All code, comments, docs in English; student-readable: plain loops, dataclasses, docstrings on every public function, a comment wherever a rule from the spec is implemented (cite the § number).
- No RL library. Allowed: numpy, gymnasium, matplotlib, imageio, imageio-ffmpeg, pytest (torch arrives in M2).
- Node indices are fixed everywhere: `M=0, H1..H5=1..5, C2=6`; facilities = 0–5; chargers = (0, 6).
- Action table is fixed: `PICK(k)=k` for k in 0–11, `DELIVER(n)=12+n` for n in 0–5, `CHARGE` = 18–23 in the order (C1,60),(C1,80),(C1,100),(C2,60),(C2,80),(C2,100), `RETURN_TO_CENTER=24`, `WAIT=25`.
- Observation layout is fixed: `[global(31) | 12 pending slots × 28 | 3 onboard slots × 16]` = 415, float32.
- Floating-point time comparisons use `EPS = 1e-9`.
- Every commit message ends with the attribution trailer given in the session (`Co-Authored-By: ...`).
- Each task ends with a short entry in `docs/CHANGELOG.md` under "M1" only if it changed something relative to the spec. After each task there is a walkthrough with Jihang, so keep functions short enough to explain.

## Review Focus

1. **High load with a full pool** (λ ≫ high, pool hits 30): rejections are counted, the ranking still returns ≤ 12 slots, nothing crashes → Task 10 `test_overload_episode_runs_and_counts_rejections`.
2. **Reward bookkeeping**: over a whole episode, the sum of step rewards must equal Σ per-order outcome rewards − 0.001 × energy used (each order counted exactly once) → Task 10 `test_reward_equals_sum_of_order_outcomes`.
3. **Expired cargo carried to C2** (not a facility) must not be disposed there, only at the next facility → Task 7 `test_expired_cargo_not_disposed_at_c2`.
4. **Same-timestamp events** (unload completing exactly at expiry; release at the same instant) follow "unload → expiry → release" → Task 7 `test_unload_at_exact_expiry_is_late_delivery`, Task 6 `test_expiry_before_release_same_time`.
5. **Stepping after termination / masked actions from `check_env`**: masked action → WAIT + `info["invalid_action"]`; stepping after `terminated` raises `RuntimeError` → Task 10 tests.

---

## File Structure

```
medical_uav/                      # repo root (/home/caijihang/projects/medical_uav)
  pyproject.toml                  # package metadata, pytest config
  environment.yml                 # exact conda environment
  README.md
  .gitignore
  docs/CHANGELOG.md               # per-milestone "what changed vs. proposal/spec"
  medical_uav/
    __init__.py
    env/
      __init__.py                 # exports MedicalUAVEnv, EnvConfig
      config.py                   # EnvConfig, OrderType, ORDER_TYPES, node constants   (Task 2)
      physics.py                  # flight time/energy, charge minutes                   (Task 2)
      astar.py                    # grid A*                                              (Task 3)
      map_gen.py                  # CityMap, generate_map, is_valid_map                  (Task 4)
      orders.py                   # Order, demand_multiplier, generate_orders            (Task 5)
      simulator.py                # Segment, Simulator (event engine + commands)        (Tasks 6–7)
      actions.py                  # action table helpers                                 (Task 8)
      masks.py                    # compute_action_mask                                  (Task 8)
      observation.py              # rank_pending, features, build_observation            (Task 9)
      uav_env.py                  # MedicalUAVEnv (Gymnasium wrapper)                    (Task 10)
      render.py                   # Renderer (replay-based matplotlib drawing)           (Task 12)
    metrics.py                    # episode_metrics                                      (Task 11)
    baselines/
      __init__.py
      heuristics.py               # random_policy, charging_rule, edf_policy             (Task 11)
  scripts/
    demo.py                       # run one episode + save animation                    (Task 12)
    calibrate.py                  # λ grid, map regeneration rate, throughput            (Task 13)
  tests/
    conftest.py                   # line_map fixture, make_sim helper
    test_setup.py, test_physics.py, test_astar.py, test_map_gen.py, test_orders.py,
    test_simulator_core.py, test_simulator_actions.py, test_masks.py,
    test_observation.py, test_env.py, test_heuristics.py, test_render.py
  docs/reports/m1/                # figures, calibration table, report outline (Task 14)
  outputs/                        # generated gifs/csv (git-ignored)
```

**Shared test fixture (defined in Task 6, used from Task 6 on):** the *line map*. Nodes lie on a line at x = [0, 2, 3, 4, 5, 6, 3.5] km (M, H1–H5, C2), y = 0; distances are |xᵢ − xⱼ|; paths are straight 2-point polylines. It makes hand-computed expectations easy.

---

### Task 1: Project setup

**Files:**
- Create: `pyproject.toml`, `environment.yml`, `.gitignore`, `README.md` (stub), `docs/CHANGELOG.md`, `medical_uav/__init__.py`, `medical_uav/env/__init__.py` (empty for now), `medical_uav/baselines/__init__.py`, `tests/test_setup.py`

**Interfaces:**
- Produces: installable package `medical_uav` (`pip install -e .`), conda env `medical-uav`.

- [ ] **Step 1: Initialise git and conda**

```bash
cd /home/caijihang/projects/medical_uav
git init -b main
conda create -y -n medical-uav python=3.11
conda run -n medical-uav pip install numpy gymnasium matplotlib imageio imageio-ffmpeg pytest
```

- [ ] **Step 2: Write `environment.yml` with exact versions**

Run `conda run -n medical-uav pip freeze`. Write `environment.yml` with `name: medical-uav`, `channels: [conda-forge]`, `dependencies: [python=3.11.*, pip, {pip: [<the six packages pinned with ==exact versions from the freeze>]}]`. Pin only the six direct dependencies.

- [ ] **Step 3: Write `pyproject.toml`**

setuptools build backend, `name = "medical_uav"`, `version = "0.1.0"`, `requires-python = ">=3.11"`, `packages = find: include medical_uav*`, `[tool.pytest.ini_options] testpaths = ["tests"]`. Then `conda run -n medical-uav pip install -e .`.

- [ ] **Step 4: Write the smoke test**

```python
# tests/test_setup.py
def test_package_imports():
    import medical_uav, gymnasium, numpy, matplotlib, imageio  # noqa: F401
```

- [ ] **Step 5: Run it**

Run: `conda run -n medical-uav python -m pytest tests/test_setup.py -v` → Expected: PASS.

- [ ] **Step 6: Commit**

`.gitignore`: `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `outputs/`, `.ipynb_checkpoints/`. `docs/CHANGELOG.md` starts with a "## M1 (Gym environment)" heading and one line pointing to the spec. README stub: project title, one-line description, "see docs/superpowers/specs/…".

```bash
git add -A && git commit -m "chore: project skeleton, conda environment, test setup"
```

---

### Task 2: Configuration and physics

**Files:**
- Create: `medical_uav/env/config.py`, `medical_uav/env/physics.py`, `tests/test_physics.py`

**Interfaces:**
- Produces:
  - Constants `M, H1, H2, H3, H4, H5, C2 = 0..6`, `NODE_NAMES = ("M","H1","H2","H3","H4","H5","C2")`, `FACILITIES = (0,1,2,3,4,5)`, `HOSPITALS = (1,2,3,4,5)`, `CHARGERS = (0, 6)`, `EPS = 1e-9`.
  - `@dataclass(frozen=True) OrderType(name: str, route: str, weight_range: tuple[float, float], limit: float, expiry: float, share: float)`. `route` is one of `"center_to_hospital"`, `"hospital_to_center"`, `"hospital_to_hospital"`.
  - `ORDER_TYPES: tuple[OrderType, ...]` = plasma, medicine, sample, supplies, with the exact §4.3 values (weights, L, H = 2L, shares 0.2/0.3/0.3/0.2).
  - `@dataclass EnvConfig` with fields and defaults: `grid_size=16, cell_km=0.25, n_zones_range=(2, 4), min_node_spacing_km=0.75, c2_min_dist_from_m_km=1.0, map_seed=0, max_map_tries=100, speed_km_per_min=1.0, energy_per_km=3.5, payload_factor=0.25, reserve=15.0, charge_rate=5.0, handling_min=2.0, max_boxes=3, max_weight=2.0, release_window=480.0, peak_windows=((60.0, 180.0), (300.0, 390.0)), peak_multiplier=2.0, offpeak_multiplier=0.6, loads={"low": 0.04, "medium": 0.07, "high": 0.10}, pool_cap=30, n_slots=12, reward_on_time=1.0, expiry_penalty=0.5, energy_cost=0.001, wait_max=10.0, charge_targets=(60.0, 80.0, 100.0)`.
  - `physics.flight_time(dist_km: float, cfg: EnvConfig) -> float`
  - `physics.flight_energy(dist_km: float, load_kg: float, cfg: EnvConfig) -> float` = dist × 3.5 × (1 + 0.25 × load)
  - `physics.charge_minutes(battery: float, target: float, cfg: EnvConfig) -> float` = max(0, target − battery) / 5
  - `physics.delivery_reward(age: float, limit: float, expiry: float, cfg: EnvConfig) -> float` (§7: +1 if age ≤ L + EPS, else 1 − 1.5·(age − L)/(H − L)).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_physics.py
import pytest
from medical_uav.env.config import EnvConfig, ORDER_TYPES
from medical_uav.env import physics

cfg = EnvConfig()

def test_order_types_match_spec():
    names = [o.name for o in ORDER_TYPES]
    assert names == ["plasma", "medicine", "sample", "supplies"]
    assert [o.limit for o in ORDER_TYPES] == [30, 90, 60, 75]
    assert [o.expiry for o in ORDER_TYPES] == [60, 180, 120, 150]
    assert sum(o.share for o in ORDER_TYPES) == pytest.approx(1.0)

def test_flight_energy_and_time():
    assert physics.flight_energy(1.0, 2.0, cfg) == pytest.approx(5.25)
    assert physics.flight_energy(2.0, 0.0, cfg) == pytest.approx(7.0)
    assert physics.flight_time(2.5, cfg) == pytest.approx(2.5)

def test_charge_minutes():
    assert physics.charge_minutes(40, 80, cfg) == pytest.approx(8.0)
    assert physics.charge_minutes(90, 80, cfg) == 0.0

@pytest.mark.parametrize("age,expected", [(10, 1.0), (30, 1.0), (45, 0.25), (60, -0.5)])
def test_delivery_reward_plasma(age, expected):
    assert physics.delivery_reward(age, 30, 60, cfg) == pytest.approx(expected)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_physics.py -v` → Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `config.py` and `physics.py`** with the interfaces above. Each spec number gets a comment citing §4.2/§4.3/§7.
- [ ] **Step 4: Run** → Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): configuration constants and physics helpers"`

---

### Task 3: Grid A*

**Files:**
- Create: `medical_uav/env/astar.py`, `tests/test_astar.py`

**Interfaces:**
- Produces:
  - `astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None` — cells as (row, col); returns the cell path including start and goal, or `None` if unreachable.
  - `path_cost_cells(path: list[tuple[int, int]]) -> float` — sum of step costs (1 orthogonal, √2 diagonal).
- Rules: 8-connected; a diagonal step is allowed only if **both** orthogonal neighbours it passes are free (no corner cutting); octile-distance heuristic; `heapq` priority queue.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_astar.py
import math
import numpy as np
import pytest
from medical_uav.env.astar import astar, path_cost_cells

def test_open_grid_diagonal():
    blocked = np.zeros((5, 5), dtype=bool)
    path = astar(blocked, (0, 0), (4, 4))
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    assert path_cost_cells(path) == pytest.approx(4 * math.sqrt(2))

def test_detour_around_wall():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0:4, 2] = True                       # wall with a gap at row 4
    path = astar(blocked, (0, 0), (0, 4))
    assert all(not blocked[r, c] for r, c in path)
    assert (4, 2) in path
    assert path_cost_cells(path) > 4

def test_no_corner_cutting():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    path = astar(blocked, (0, 0), (1, 1))
    assert path == [(0, 0), (1, 0), (1, 1)]

def test_unreachable_returns_none():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[:, 1] = True
    assert astar(blocked, (0, 0), (0, 2)) is None
```

- [ ] **Step 2: Run** `python -m pytest tests/test_astar.py -v` → FAIL.
- [ ] **Step 3: Implement** `astar` and `path_cost_cells` (~50 lines, commented step by step: open set, g-score, came-from, reconstruct).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): 8-connected A* without corner cutting"`

---

### Task 4: Map generation and validity

**Files:**
- Create: `medical_uav/env/map_gen.py`, `tests/test_map_gen.py`

**Interfaces:**
- Consumes: `astar`, `path_cost_cells` (Task 3); `EnvConfig`, node constants, `physics.flight_energy` (Task 2).
- Produces:
  - `@dataclass CityMap(node_xy: np.ndarray (7,2) km, dist_km: np.ndarray (7,7), paths: dict[tuple[int,int], np.ndarray (k,2) km polyline], blocked: np.ndarray (G,G) bool | None, zones: list[tuple[int,int,int,int]] (row, col, height, width), seed: int, tries: int)`.
  - `CityMap.from_positions(node_xy: np.ndarray) -> CityMap` (classmethod; Euclidean distances, straight 2-point paths, `blocked=None`, `zones=[]`, `seed=-1`, `tries=0`) — used by tests and the line map.
  - `generate_map(seed: int, cfg: EnvConfig) -> CityMap`
  - `is_valid_map(dist_km: np.ndarray, cfg: EnvConfig) -> bool`
- Generation (one attempt, all randomness from `np.random.default_rng(seed)`, attempts continue with the same rng): sample `n_zones` in `n_zones_range` rectangles with height and width 2–5 cells → place 7 nodes on free cells (uniform random, rejection until spacing ≥ 3 cells = 0.75 km between all nodes and C2 ≥ 4 cells from M; give up the attempt after 500 rejections) → A* for all 21 pairs (fail if any is `None`) → `dist_km = path_cost_cells × cell_km` → paths converted to km polylines through cell centres → `is_valid_map`. Raise `RuntimeError` after `max_map_tries` attempts. `tries` = attempts used.
- Validity (§4.1): for every charger c, every node i, every node j: `100 − E(c→i, 2) − E(i→j, 2) − min_c' E(j→c', 2) ≥ 15`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_map_gen.py
import numpy as np
from medical_uav.env.config import EnvConfig, M, C2
from medical_uav.env.map_gen import generate_map, is_valid_map, CityMap

cfg = EnvConfig()

def test_deterministic():
    a, b = generate_map(0, cfg), generate_map(0, cfg)
    assert np.array_equal(a.dist_km, b.dist_km) and a.zones == b.zones

def test_distances_symmetric_and_at_least_euclidean():
    m = generate_map(1, cfg)
    assert np.allclose(m.dist_km, m.dist_km.T)
    for i in range(7):
        for j in range(7):
            assert m.dist_km[i, j] >= np.linalg.norm(m.node_xy[i] - m.node_xy[j]) - 1e-9

def test_spacing_rules():
    m = generate_map(2, cfg)
    for i in range(7):
        for j in range(i + 1, 7):
            assert np.linalg.norm(m.node_xy[i] - m.node_xy[j]) >= 0.75 - 1e-9
    assert np.linalg.norm(m.node_xy[M] - m.node_xy[C2]) >= 1.0 - 1e-9

def test_generated_maps_are_valid_and_paths_avoid_zones():
    for seed in range(20):
        m = generate_map(seed, cfg)
        assert is_valid_map(m.dist_km, cfg)
        assert m.paths[(0, 1)].shape[1] == 2

def test_invalid_when_too_far():
    far = CityMap.from_positions(np.array([[0, 0]] + [[20, 0]] * 5 + [[1, 0]], dtype=float))
    assert not is_valid_map(far.dist_km, cfg)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `map_gen.py` as specified.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): random city map with no-fly zones, A* distance matrix, validity check"`

---

### Task 5: Order stream generator

**Files:**
- Create: `medical_uav/env/orders.py`, `tests/test_orders.py`

**Interfaces:**
- Consumes: `EnvConfig`, `ORDER_TYPES`, node constants (Task 2).
- Produces:
  - `@dataclass Order(id: int, type_idx: int, origin: int, dest: int, weight: float, release: float, limit: float, expiry_after: float, status: str = "future", pickup_time: float | None = None, end_time: float | None = None, disposed_time: float | None = None)`; `status` ∈ {"future","pending","onboard","delivered","expired","rejected"}; properties `deadline = release + limit`, `expiry = release + expiry_after`; method `age(t) -> float`.
  - `demand_multiplier(t: float, cfg: EnvConfig) -> float` — 2.0 inside a peak window, 0.6 elsewhere inside [0, 480), 0.0 outside.
  - `generate_orders(demand_seed: int, lam: float, cfg: EnvConfig) -> list[Order]` — piecewise segments [0,60),[60,180),[180,300),[300,390),[390,480); per segment `n ~ Poisson(lam × m × length)`, release times uniform within the segment; sort by release; assign ids 0..n−1 in that order; type by `share`; origin/dest by `route` (hospitals uniform; supplies dest ≠ origin); weight uniform in range. Single `np.random.default_rng(demand_seed)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orders.py
import numpy as np, pytest
from medical_uav.env.config import EnvConfig, M, HOSPITALS
from medical_uav.env.orders import generate_orders, demand_multiplier

cfg = EnvConfig()

def test_multiplier_profile():
    assert demand_multiplier(30, cfg) == 0.6
    assert demand_multiplier(60, cfg) == 2.0
    assert demand_multiplier(179.9, cfg) == 2.0
    assert demand_multiplier(200, cfg) == 0.6
    assert demand_multiplier(480, cfg) == 0.0

def test_deterministic_and_sorted():
    a, b = generate_orders(5, 0.07, cfg), generate_orders(5, 0.07, cfg)
    assert [(o.release, o.origin, o.dest, o.weight) for o in a] == [(o.release, o.origin, o.dest, o.weight) for o in b]
    assert all(a[i].release <= a[i + 1].release for i in range(len(a) - 1))
    assert [o.id for o in a] == list(range(len(a)))

def test_expected_count_is_582_lambda():
    counts = [len(generate_orders(s, 0.07, cfg)) for s in range(400)]
    assert np.mean(counts) == pytest.approx(582 * 0.07, rel=0.05)

def test_routes_weights_limits():
    for o in generate_orders(7, 0.2, cfg):
        name = ["plasma", "medicine", "sample", "supplies"][o.type_idx]
        if name in ("plasma", "medicine"):
            assert o.origin == M and o.dest in HOSPITALS
        elif name == "sample":
            assert o.origin in HOSPITALS and o.dest == M
        else:
            assert o.origin in HOSPITALS and o.dest in HOSPITALS and o.origin != o.dest
        assert 0.2 <= o.weight <= 0.9 and 0 <= o.release < 480
        assert o.expiry_after == 2 * o.limit
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): seeded non-homogeneous Poisson order generator"`

---

### Task 6: Simulator core — clock, releases, expiries, WAIT, termination

**Files:**
- Create: `medical_uav/env/simulator.py`, `tests/conftest.py`, `tests/test_simulator_core.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5.
- Produces:
  - `@dataclass Segment(kind: str, t0: float, t1: float, node_from: int, node_to: int, battery0: float, battery1: float, order_id: int | None = None)`; `kind` ∈ {"fly","load","unload","dispose","charge","wait"}.
  - `class Simulator`, constructed as `Simulator(cfg: EnvConfig, city_map: CityMap, orders: list[Order], lam: float)`. Public state: `cfg`, `city_map`, `lam`, `t`, `location`, `battery`, `onboard: list[Order]` (includes expired cargo), `pending: list[Order]`, `orders`, `segments: list[Segment]`, `total_reward`, `energy_used`, `km_flown`, `charging_time`, `release_times_log: list[float]`, `events: list[tuple[float, str, int]]` — an outcome log of `(time, kind, order_id)` with kind ∈ {"release","reject","expire","pickup","deliver","dispose"}, appended whenever that outcome happens (Tasks 6–7).
  - Methods in this task: `load_kg() -> float` (all onboard, incl. expired cargo); `energy(a: int, b: int, load: float) -> float`; `travel_time(a: int, b: int) -> float`; `nearest_charger(node: int, load: float) -> tuple[int, float]` (charger, energy; ties → lower index); `next_event_time() -> float` (next release or next expiry of any pending/non-expired onboard order, `inf` if none); `advance_to(t_target: float, strict: bool = False) -> None`; `do_wait() -> None`; `is_done() -> bool`; `check_invariant() -> None`.
- `advance_to` rule (§4.4): repeatedly take the earliest pending event with time < t_target (strict) or ≤ t_target + EPS (non-strict); at equal times process **expiries before releases**. Expiry → status "expired", `end_time`, reward −0.5, removed from `pending` (onboard boxes stay in `onboard`). Release → if `len(pending) >= pool_cap` status "rejected" + reward −0.5, else "pending"; log release time. Finally set `t = t_target`. (Unload completions come first because Task 7's deliver calls `advance_to(..., strict=True)` before recording an unload.)
- `do_wait`: duration = min(next_event_time, t + 10) − t (always > 0); appends a "wait" segment; `advance_to(t + duration)`.
- `is_done`: `t >= release_window − EPS` and every order is no longer "future"/"pending" and no onboard box has status "onboard".
- `check_invariant`: `battery >= reserve + nearest_charger(location, load_kg())[1] − 1e-6`, else `AssertionError`.
- Constructor: start at M, battery 100, then `advance_to(0.0)` so orders released at t = 0 are pending.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import numpy as np, pytest
from medical_uav.env.config import EnvConfig
from medical_uav.env.map_gen import CityMap
from medical_uav.env.orders import Order
from medical_uav.env.simulator import Simulator

LINE_X = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5]   # M, H1..H5, C2

@pytest.fixture
def line_map():
    return CityMap.from_positions(np.array([[x, 0.0] for x in LINE_X]))

def make_order(oid, type_idx, origin, dest, weight, release):
    limits = [30.0, 90.0, 60.0, 75.0]
    return Order(oid, type_idx, origin, dest, weight, release, limits[type_idx], 2 * limits[type_idx])

def make_sim(city_map, orders, cfg=None):
    return Simulator(cfg or EnvConfig(), city_map, orders, lam=0.07)
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_simulator_core.py
import pytest
from medical_uav.env.config import EnvConfig, M, C2
from tests.conftest import make_order, make_sim

def test_start_state(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    assert sim.t == 0 and sim.location == M and sim.battery == 100
    assert [o.id for o in sim.pending] == [0]

def test_wait_stops_at_next_release(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 1, 0.5, 4.0)])
    sim.do_wait()
    assert sim.t == pytest.approx(4.0) and len(sim.pending) == 1

def test_wait_max_ten_minutes(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 1, 0.5, 100.0)])
    sim.do_wait()
    assert sim.t == pytest.approx(10.0)

def test_pending_order_expires_with_penalty(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])   # plasma, H = 60
    sim.advance_to(60.0)
    o = sim.orders[0]
    assert o.status == "expired" and o.end_time == pytest.approx(60.0)
    assert sim.pending == [] and sim.total_reward == pytest.approx(-0.5)

def test_expiry_before_release_same_time(line_map):
    cfg = EnvConfig(pool_cap=1)
    orders = [make_order(0, 0, M, 1, 0.7, 0.0), make_order(1, 0, M, 2, 0.7, 60.0)]
    sim = make_sim(line_map, orders, cfg)
    sim.advance_to(60.0)
    assert orders[0].status == "expired" and orders[1].status == "pending"

def test_pool_cap_rejects(line_map):
    cfg = EnvConfig(pool_cap=2)
    orders = [make_order(i, 1, M, 1, 0.5, 5.0) for i in range(3)]
    sim = make_sim(line_map, orders, cfg)
    sim.advance_to(5.0)
    assert [o.status for o in orders] == ["pending", "pending", "rejected"]
    assert sim.total_reward == pytest.approx(-0.5)

def test_done_only_after_release_window(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    sim.advance_to(61.0)
    assert not sim.is_done()
    sim.advance_to(480.0)
    assert sim.is_done()

def test_nearest_charger(line_map):
    sim = make_sim(line_map, [])
    assert sim.nearest_charger(5, 0.0) == (C2, pytest.approx(2.5 * 3.5))   # H5 -> C2 is 2.5 km
    assert sim.nearest_charger(1, 0.0) == (C2, pytest.approx(1.5 * 3.5))   # H1 -> C2 1.5 km < H1 -> M 2 km
    assert sim.nearest_charger(M, 0.0) == (M, 0.0)
```

- [ ] **Step 3: Run** `python -m pytest tests/test_simulator_core.py -v` → FAIL.
- [ ] **Step 4: Implement** the Task 6 part of `Simulator`.
- [ ] **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat(env): event-driven simulator core (releases, expiries, wait, termination)"`

---

### Task 7: Simulator commands — fly, dispose, pick, deliver, charge, return

**Files:**
- Modify: `medical_uav/env/simulator.py`
- Create: `tests/test_simulator_actions.py`

**Interfaces:**
- Consumes: Task 6 `Simulator`, `physics.delivery_reward`.
- Produces (all mutate state, add reward to `total_reward`, append `Segment`s). The safety invariant is checked only at macro-action boundaries: `do_pick`, `do_deliver`, `do_charge`, `do_return` and `do_wait` end with `check_invariant()`; the helpers `fly_to` and `dispose_expired_cargo` do not (a loaded flight may legitimately be below the invariant until the unload finishes):
  - `fly_to(node: int) -> None` — if `node == location` do nothing; otherwise duration `travel_time`, energy `energy(location, node, load_kg())` subtracted, reward −0.001 × energy, `energy_used`/`km_flown` updated, "fly" segment, `advance_to(arrival)`, `location = node`; then, **if `node` is a facility (0–5)**, call `dispose_expired_cargo()`.
  - `dispose_expired_cargo() -> None` — each onboard box with status "expired": 2 min "dispose" segment, `advance_to(end)`, `disposed_time` set, removed from `onboard`. No reward.
  - `do_pick(order: Order) -> None` — `fly_to(order.origin)`; 2 min "load" segment; `advance_to(t + 2)`; status "onboard", `pickup_time`, moved from `pending` to `onboard`. (Masks guarantee it completes before expiry; assert it.)
  - `do_deliver(node: int) -> None` — `fly_to(node)`; then for each onboard box with `dest == node`, most urgent first (deadline, then id): if status is "expired", dispose it (2 min, no reward); else `advance_to(t_complete, strict=True)` then record delivery at `t_complete = t + 2`: status "delivered", `end_time`, reward `delivery_reward(age)`; after the loop `advance_to(t)` (non-strict) to process expiries/releases at that instant.
  - `do_charge(station: int, target: float) -> None` — `fly_to(station)`; minutes = `charge_minutes(battery, target)`; "charge" segment; `advance_to(t + minutes)`; battery = target; `charging_time` += minutes.
  - `do_return() -> None` — `fly_to(M)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_simulator_actions.py
import pytest
from medical_uav.env.config import M, C2
from tests.conftest import make_order, make_sim

def test_pick_and_on_time_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.8, 0.0)])
    sim.do_pick(sim.orders[0])
    assert sim.t == pytest.approx(2.0) and sim.orders[0].status == "onboard"
    r0 = sim.total_reward
    sim.do_deliver(1)                               # 2 km with 0.8 kg
    energy = 2 * 3.5 * (1 + 0.25 * 0.8)             # 8.4
    assert sim.t == pytest.approx(6.0)
    assert sim.battery == pytest.approx(100 - energy)
    assert sim.total_reward - r0 == pytest.approx(1.0 - 0.001 * energy)
    assert sim.orders[0].status == "delivered" and sim.onboard == []

def test_late_delivery_reward(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.8, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.advance_to(41.0)                            # arrive 43, unload done 45 -> age 45
    r0 = sim.total_reward
    sim.do_deliver(1)
    assert sim.total_reward - r0 == pytest.approx(0.25 - 0.001 * 8.4)

def test_unload_at_exact_expiry_is_late_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 4, 0.6, 0.0)])   # plasma to H4 (5 km)
    sim.do_pick(sim.orders[0])
    sim.advance_to(53.0)                            # arrive 58, unload done 60 = expiry
    sim.do_deliver(4)
    assert sim.orders[0].status == "delivered" and sim.orders[0].end_time == pytest.approx(60.0)

def test_two_boxes_same_destination_unload_sequentially(line_map):
    orders = [make_order(0, 1, M, 2, 0.5, 0.0), make_order(1, 0, M, 2, 0.7, 0.0)]
    sim = make_sim(line_map, orders)
    sim.do_pick(orders[0]); sim.do_pick(orders[1])          # t = 4, both at M (no flight)
    sim.do_deliver(2)                                       # 3 km, arrive 7
    assert orders[1].end_time == pytest.approx(9.0)         # plasma first (earlier deadline)
    assert orders[0].end_time == pytest.approx(11.0)

def test_expired_cargo_not_disposed_at_c2(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 5, 0.6, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.advance_to(60.0)                                    # box expires on board
    assert sim.orders[0].status == "expired" and len(sim.onboard) == 1
    sim.do_charge(C2, 100.0)
    assert len(sim.onboard) == 1                            # C2 is not a facility
    sim.do_return()
    assert sim.onboard == [] and sim.orders[0].disposed_time is not None

def test_charge_to_target(line_map):
    sim = make_sim(line_map, [])
    sim.battery = 50.0
    sim.do_charge(M, 80.0)
    assert sim.battery == pytest.approx(80.0) and sim.t == pytest.approx(6.0)
    assert sim.charging_time == pytest.approx(6.0)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `python -m pytest tests/test_simulator_core.py tests/test_simulator_actions.py -v` → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): pick/deliver/charge/return commands with expired-cargo disposal"`

---

### Task 8: Action table and masks

**Files:**
- Create: `medical_uav/env/actions.py`, `medical_uav/env/masks.py`, `tests/test_masks.py`

**Interfaces:**
- Consumes: `Simulator` (Tasks 6–7).
- Produces:
  - `actions.py`: `N_ACTIONS = 26`, `PICK0 = 0`, `DELIVER0 = 12`, `CHARGE0 = 18`, `RETURN = 24`, `WAIT = 25`, `CHARGE_TABLE = ((0, 60.0), (0, 80.0), (0, 100.0), (6, 60.0), (6, 80.0), (6, 100.0))`, `decode(a: int) -> tuple[str, int | tuple[int, float] | None]` (kinds "pick","deliver","charge","return","wait"), `action_name(a: int) -> str` (e.g. `"PICK(3)"`, `"DELIVER(H2)"`, `"CHARGE(C2,80)"`).
  - `masks.py`: `predict_arrival(sim, node: int) -> tuple[float, list[Order]]` — arrival time and the onboard boxes that will be expired cargo on arrival (expiry ≤ arrival + EPS or already expired); `compute_action_mask(sim, slots: list[Order]) -> np.ndarray` (bool, shape (26,)) implementing the §5 rules exactly, where disposal at the target happens only if the target is a facility.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_masks.py
import numpy as np, pytest
from medical_uav.env.config import EnvConfig, M, C2
from medical_uav.env.actions import decode, action_name, N_ACTIONS, WAIT, RETURN, DELIVER0, CHARGE0
from medical_uav.env.masks import compute_action_mask
from tests.conftest import make_order, make_sim

def test_decode_table():
    assert decode(3) == ("pick", 3) and decode(DELIVER0 + 2) == ("deliver", 2)
    assert decode(CHARGE0 + 4) == ("charge", (C2, 80.0))
    assert decode(24) == ("return", None) and decode(25) == ("wait", None)
    assert action_name(CHARGE0 + 4) == "CHARGE(C2,80)"

def test_wait_always_valid_and_return_masked_at_center(line_map):
    sim = make_sim(line_map, [])
    mask = compute_action_mask(sim, [])
    assert mask.shape == (N_ACTIONS,) and mask[WAIT] and not mask[RETURN]

def test_capacity_boxes_and_weight(line_map):
    orders = [make_order(i, 1, M, 1, 0.4, 0.0) for i in range(3)] + [make_order(3, 2, 2, M, 0.3, 0.0)]
    sim = make_sim(line_map, orders)
    for o in orders[:3]:
        sim.do_pick(o)                              # 3 boxes, 1.2 kg: only the box limit binds
    assert not compute_action_mask(sim, [orders[3]])[0]

def test_weight_limit(line_map):
    a, b = make_order(0, 0, M, 1, 0.9, 0.0), make_order(1, 0, M, 2, 0.9, 0.0)
    c = make_order(2, 1, M, 3, 0.6, 0.0)
    sim = make_sim(line_map, [a, b, c])
    sim.do_pick(a); sim.do_pick(b)                  # 1.8 kg
    assert not compute_action_mask(sim, [c])[0]     # 2.4 kg > 2.0

def test_pick_masked_if_cannot_load_before_expiry(line_map):
    o = make_order(0, 2, 5, M, 0.3, 0.0)            # sample at H5 (6 km away), expires at 120
    sim = make_sim(line_map, [o])
    sim.advance_to(113.0)                           # arrive 119, load done 121 > 120
    assert not compute_action_mask(sim, [o])[0]

def test_pick_masked_when_battery_too_low(line_map):
    o = make_order(0, 2, 5, M, 0.4, 0.0)
    sim = make_sim(line_map, [o])
    sim.location, sim.battery = 5, 20.0             # at H5; to C2 with 0.4 kg costs 9.625
    assert not compute_action_mask(sim, [o])[0]

def test_charge_targets(line_map):
    sim = make_sim(line_map, [])
    sim.battery = 100.0
    # at M nothing to charge; C2 arrival = 100 - 12.25 = 87.75, so only (C2, 100) is valid
    assert list(compute_action_mask(sim, [])[CHARGE0:CHARGE0 + 6]) == [False, False, False, False, False, True]
    sim.battery = 50.0
    m = compute_action_mask(sim, [])
    assert m[CHARGE0 + 0] and m[CHARGE0 + 1] and m[CHARGE0 + 2]      # C1 60/80/100
    assert m[CHARGE0 + 3]                                             # C2: arrival 37.75, 60 > 42.75

def test_expired_cargo_frees_capacity_at_origin(line_map):
    old = [make_order(i, 0, M, 1, 0.6, 0.0) for i in range(3)]       # plasma, expire at 60
    new = make_order(3, 1, M, 2, 0.5, 55.0)
    sim = make_sim(line_map, old + [new])
    for o in old:
        sim.do_pick(o)
    sim.do_charge(C2, 100.0); sim.advance_to(57.0)                    # at C2, still 3 boxes
    assert compute_action_mask(sim, [new])[0]       # arrive at M 60.5 >= expiry 60: disposed, slot frees
```
The random-rollout invariant property test lives in Task 10 (`tests/test_env.py`), once the full environment exists.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `actions.py` and `masks.py` (one small helper per action kind, each citing the §5 rule).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): action table and safety-preserving action masks"`

---

### Task 9: Observation

**Files:**
- Create: `medical_uav/env/observation.py`, `tests/test_observation.py`

**Interfaces:**
- Consumes: `Simulator`, `demand_multiplier`, `compute_action_mask`.
- Produces:
  - Constants `GLOBAL_DIM = 31`, `PENDING_DIM = 28`, `N_SLOTS = 12`, `ONBOARD_DIM = 16`, `N_ONBOARD = 3`, `OBS_DIM = 415`.
  - `direct_completion_time(sim, order: Order) -> float` — pending: travel(here→origin) + 2 + travel(origin→dest) + 2; onboard: travel(here→dest) + 2.
  - `provisional_route_time(sim, extra: Order | None = None) -> float` — §6 definition (visit destinations of non-expired onboard boxes, plus `extra`'s destination, in earliest-deadline order of the boxes; with `extra`: go to its origin first and add 2 min load; add 2 min unload per box).
  - `insertion_extra_time(sim, order) -> float` = `provisional_route_time(sim, order) − provisional_route_time(sim)`.
  - `rank_pending(sim, n_slots: int = 12) -> list[Order]` — the two-tier ranking of §6.
  - `build_observation(sim, slots: list[Order], mask: np.ndarray) -> np.ndarray` — float32 (415,), features in exactly the order of the §6 table (global block in the listed order; per slot: valid, type one-hot(4), origin one-hot(6), dest one-hot(6), weight, ttd/L, tte/L, age/L, direct/30, slack/30, same-dest flag, n-same-dest/5, origin-is-here, insertion/30, pick-feasible = `mask[k]`). Onboard slots sorted by deadline then id; expired cargo included with the expired flag.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_observation.py
import numpy as np, pytest
from medical_uav.env.config import M
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import (OBS_DIM, GLOBAL_DIM, PENDING_DIM, rank_pending,
                                         insertion_extra_time, build_observation)
from tests.conftest import make_order, make_sim

def test_two_tier_ranking_puts_on_time_feasible_first(line_map):
    late = make_order(0, 0, M, 1, 0.7, 0.0)          # plasma: deadline 30, expiry 60
    fresh = make_order(1, 1, M, 2, 0.5, 40.0)        # medicine: deadline 130
    sim = make_sim(line_map, [late, fresh])
    sim.advance_to(50.0)
    assert [o.id for o in rank_pending(sim)] == [1, 0]

def test_at_most_twelve_slots(line_map):
    orders = [make_order(i, 1, M, 1 + i % 5, 0.4, 0.0) for i in range(20)]
    sim = make_sim(line_map, orders)
    assert len(rank_pending(sim)) == 12

def test_insertion_extra_time(line_map):
    a, b = make_order(0, 1, M, 1, 0.4, 0.0), make_order(1, 1, M, 1, 0.4, 0.0)
    sim = make_sim(line_map, [a, b])
    assert insertion_extra_time(sim, a) == pytest.approx(6.0)     # 0 + 2 load + 2 fly + 2 unload
    sim.do_pick(a)
    assert insertion_extra_time(sim, b) == pytest.approx(4.0)     # 2 load + 2 extra unload

def test_observation_shape_and_slot_encoding(line_map):
    o = make_order(0, 0, M, 1, 0.7, 0.0)
    sim = make_sim(line_map, [o])
    slots = rank_pending(sim)
    obs = build_observation(sim, slots, compute_action_mask(sim, slots))
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    s0 = obs[GLOBAL_DIM:GLOBAL_DIM + PENDING_DIM]
    assert s0[0] == 1.0 and s0[1] == 1.0                            # valid, type plasma
    assert obs[GLOBAL_DIM + PENDING_DIM] == 0.0                     # slot 1 invalid (padding)

def test_late_order_has_negative_time_to_deadline(line_map):
    o = make_order(0, 0, M, 1, 0.7, 0.0)
    sim = make_sim(line_map, [o])
    sim.advance_to(45.0)
    slots = rank_pending(sim)
    obs = build_observation(sim, slots, compute_action_mask(sim, slots))
    ttd = obs[GLOBAL_DIM + 1 + 4 + 6 + 6 + 1]                      # index of ttd/L
    assert ttd == pytest.approx(-0.5)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(env): 415-dim observation with two-tier slot ranking and batching hints"`

---

### Task 10: Gymnasium environment

**Files:**
- Create: `medical_uav/env/uav_env.py`, `tests/test_env.py`
- Modify: `medical_uav/env/__init__.py` (export `MedicalUAVEnv`, `EnvConfig`)

**Interfaces:**
- Consumes: everything above.
- Produces `class MedicalUAVEnv(gymnasium.Env)`:
  - `metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}`
  - `__init__(self, config: EnvConfig | None = None, load: str = "medium", render_mode: str | None = None)`; builds the fixed map once with `generate_map(config.map_seed, config)`.
  - `observation_space = Box(-inf, inf, (415,), float32)`, `action_space = Discrete(26)`.
  - `reset(seed=None, options=None) -> (obs, info)`; `options` keys (all optional): `"demand_seed": int` (default: drawn from `self.np_random`), `"load": "low"|"medium"|"high"` (default: constructor value), `"lam": float` (overrides load). Creates a new `Simulator`, computes `self.slots = rank_pending(sim)`.
  - `step(action) -> (obs, reward, terminated, truncated, info)`: raise `RuntimeError` if the episode already terminated; if `mask[action]` is False → execute WAIT and set `info["invalid_action"] = True`; reward = change in `sim.total_reward`; `terminated = sim.is_done()`; `truncated = False`.
  - `info` keys: `"action_mask"` (bool (26,)), `"t"`, `"elapsed"`, `"invalid_action"`, `"action_name"`, `"events"` (the slice of `sim.events` appended during this step; empty list after `reset`).
  - `action_masks() -> np.ndarray`; attributes `sim`, `slots`, `city_map`, `lam`.
  - `render()` → delegates to `Renderer` (Task 12); until then returns `None`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_env.py
import numpy as np, pytest
from gymnasium.utils.env_checker import check_env
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import ORDER_TYPES
from medical_uav.env.actions import WAIT

def rollout(env, seed, rng_seed=0, options=None):
    rng = np.random.default_rng(rng_seed)
    obs, info = env.reset(seed=seed, options=options)
    total, steps = 0.0, 0
    while True:
        valid = np.flatnonzero(info["action_mask"])
        obs, r, term, trunc, info = env.step(int(rng.choice(valid)))
        env.sim.check_invariant()
        assert info["action_mask"].any()
        total += r; steps += 1
        if term or trunc:
            return total, steps

def test_check_env():
    check_env(MedicalUAVEnv(), skip_render_check=True)

def test_determinism_same_seed():
    a, b = MedicalUAVEnv(), MedicalUAVEnv()
    assert rollout(a, 3) == rollout(b, 3)

@pytest.mark.parametrize("seed", range(10))
def test_random_rollouts_keep_invariant(seed):
    rollout(MedicalUAVEnv(load="high"), seed)

def test_invalid_action_becomes_wait():
    env = MedicalUAVEnv()
    obs, info = env.reset(seed=0)
    invalid = int(np.flatnonzero(~info["action_mask"])[0])
    t0 = env.sim.t
    _, _, _, _, info = env.step(invalid)
    assert info["invalid_action"] and env.sim.t > t0

def test_info_events_report_delivery():
    env = MedicalUAVEnv(load="medium")
    obs, info = env.reset(seed=0)
    kinds = []
    while not kinds or "deliver" not in kinds:
        mask = info["action_mask"]
        a = next((a for a in range(12, 18) if mask[a]), next((a for a in range(12) if mask[a]), 25))
        obs, r, term, trunc, info = env.step(a)
        kinds += [k for (_, k, _) in info["events"]]
        assert not term
    assert "pickup" in kinds and "deliver" in kinds

def test_step_after_done_raises():
    env = MedicalUAVEnv(load="low")
    rollout(env, 1)
    with pytest.raises(RuntimeError):
        env.step(WAIT)

def test_overload_episode_runs_and_counts_rejections():
    env = MedicalUAVEnv()
    rollout(env, 2, options={"lam": 0.4})
    statuses = [o.status for o in env.sim.orders]
    assert "rejected" in statuses
    assert all(s in ("delivered", "expired", "rejected") for s in statuses)

def test_reward_equals_sum_of_order_outcomes():
    env = MedicalUAVEnv(load="high")
    total, _ = rollout(env, 4)
    from medical_uav.env.physics import delivery_reward
    expected = 0.0
    for o in env.sim.orders:
        if o.status == "delivered":
            expected += delivery_reward(o.end_time - o.release, o.limit, o.expiry_after, env.sim.cfg)
        else:
            expected -= 0.5
    expected -= 0.001 * env.sim.energy_used
    assert total == pytest.approx(expected, abs=1e-6)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `uav_env.py`.
- [ ] **Step 4: Run the full suite** `python -m pytest -q` → all PASS (check_env may print warnings; there must be no errors).
- [ ] **Step 5: Commit** `git commit -m "feat(env): Gymnasium MedicalUAVEnv with masks, info and seeding"`

---

### Task 11: Metrics, Random and EDF baselines with the shared charging rule

**Files:**
- Create: `medical_uav/metrics.py`, `medical_uav/baselines/heuristics.py`, `tests/test_heuristics.py`

**Interfaces:**
- Consumes: `MedicalUAVEnv` (Task 10).
- Produces:
  - `episode_metrics(sim) -> dict[str, float]` with keys `n_orders, on_time, on_time_rate, served_rate, expiry_rate, rejected, mean_lateness, km_flown, energy_used, charge_minutes, episode_minutes, charge_share, busy_share` (busy = fly + load + unload + dispose minutes, from `sim.segments`). Rates divide by `n_orders` (0 if no orders).
  - `random_policy(env, rng: np.random.Generator) -> int`
  - `charging_rule(env, theta: float) -> int | None` — §9 steps 1–2; returns a valid CHARGE action or `None`.
  - `edf_policy(env, theta: float = 40.0) -> int` — `charging_rule` first; else among valid PICK(k) (key `slots[k].deadline`) and valid DELIVER(n) (key = earliest deadline of non-expired onboard boxes for n) choose the smallest key (ties: DELIVER before PICK, then lower index); else WAIT.
  - `run_episode(env, policy, seed: int, options: dict | None = None) -> dict` — convenience loop returning `episode_metrics` plus `"return"` and `"decisions"`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_heuristics.py
import numpy as np, pytest
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import CHARGE0, DELIVER0
from medical_uav.baselines.heuristics import edf_policy, charging_rule, random_policy, run_episode

def test_edf_picks_earliest_deadline():
    env = MedicalUAVEnv()
    env.reset(seed=11)
    while not env.slots:                                   # wait for some orders
        env.step(25)
    a = edf_policy(env)
    valid_picks = [k for k in range(len(env.slots)) if env.action_masks()[k]]
    if a < DELIVER0 and valid_picks:
        assert env.slots[a].deadline == min(env.slots[k].deadline for k in valid_picks)

@pytest.mark.parametrize("seed", range(5))
def test_policies_only_choose_valid_actions(seed):
    env = MedicalUAVEnv(load="high")
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        a = edf_policy(env)
        assert info["action_mask"][a]
        c = charging_rule(env, 50.0)
        assert c is None or info["action_mask"][c]
        obs, r, done, _, info = env.step(a)

def test_metrics_account_for_every_order():
    env = MedicalUAVEnv()
    m = run_episode(env, edf_policy, seed=0)
    assert 0.0 <= m["on_time_rate"] <= m["served_rate"] <= 1.0
    assert m["served_rate"] + m["expiry_rate"] + m["rejected"] / max(m["n_orders"], 1) == pytest.approx(1.0)

def test_edf_beats_random_on_low_load():
    env = MedicalUAVEnv(load="low")
    rng = np.random.default_rng(0)
    edf = np.mean([run_episode(env, edf_policy, s)["on_time_rate"] for s in range(5)])
    rnd = np.mean([run_episode(env, lambda e: random_policy(e, rng), s)["on_time_rate"] for s in range(5)])
    assert edf > rnd
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: episode metrics, random and EDF baselines with shared charging rule"`

---

### Task 12: Renderer and demo script

**Files:**
- Create: `medical_uav/env/render.py`, `scripts/demo.py`, `tests/test_render.py`
- Modify: `medical_uav/env/uav_env.py` (`render()` uses `Renderer`)

**Interfaces:**
- Consumes: `Simulator.segments`, order timestamps, `CityMap` (paths, zones, node_xy).
- Produces:
  - `drone_state_at(sim, t: float) -> dict` with keys `"xy"` (np.ndarray (2,) km), `"battery"`, `"cargo"` (orders on board at t: `pickup_time ≤ t` and not yet delivered/disposed), `"pending"` (released, not picked, not ended at t), `"activity"` (segment kind at t). Position is interpolated by arc length along `city_map.paths[(from, to)]` during "fly" segments; battery is linear inside fly/charge segments.
  - `class Renderer(city_map, cfg)` with `draw(sim, t: float) -> np.ndarray` (uint8 (H, W, 3)). Layout: left = map (zones as red rectangles, nodes labelled M/H1–H5/C2 with charger rings, current flight path highlighted, drone marker); right = side panel (clock hh:mm with 08:00 = t 0, battery bar with the 15 % reserve line, 3 cargo slots coloured by order type with minutes left, up to 12 pending orders coloured green/amber/red by time-to-deadline, cumulative on-time count).
  - `save_animation(frames: list[np.ndarray], path: str, fps: int) -> None` (`.gif` or `.mp4` via imageio).
  - `MedicalUAVEnv.render()`: `"rgb_array"` → `Renderer.draw(sim, sim.t)`; `"human"` → show the same frame with `plt.pause`.
  - `scripts/demo.py` CLI: `--policy {random,edf}` (default edf), `--load {low,medium,high}` (default medium), `--seed` (default 0), `--out` (default `outputs/demo_<policy>_<load>_<seed>.gif`), `--fps` (default 12), `--minutes-per-frame` (default 1.0), `--no-video`. Runs one episode, prints the metrics table, then renders frames for t = 0, step, … up to the final time and saves the animation.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_render.py
import numpy as np, pytest
from medical_uav.env.config import M
from medical_uav.env.render import Renderer, drone_state_at
from tests.conftest import make_order, make_sim

def test_position_interpolates_along_flight(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])                       # t = 2 at M
    sim.do_deliver(2)                                 # fly 2 -> 5 (3 km)
    s = drone_state_at(sim, 3.5)
    assert s["xy"] == pytest.approx(np.array([1.5, 0.0]))
    assert s["activity"] == "fly" and len(s["cargo"]) == 1

def test_cargo_empty_after_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0]); sim.do_deliver(2)
    assert drone_state_at(sim, sim.t)["cargo"] == []

def test_draw_returns_rgb_frame(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    frame = Renderer(line_map, sim.cfg).draw(sim, 0.0)
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `render.py`, wire `render()`, write `scripts/demo.py`.
- [ ] **Step 4: Run** `python -m pytest tests/test_render.py -v` → PASS; then `python scripts/demo.py --policy edf --load medium --seed 0` → prints metrics and writes `outputs/demo_edf_medium_0.gif`. Open the gif and check visually: the drone follows the A* paths around no-fly zones, cargo slots fill and empty, the battery drops in flight and rises while charging.
- [ ] **Step 5: Commit** `git commit -m "feat(env): replay-based matplotlib renderer and demo script"`

---

### Task 13: Calibration and throughput (gate G1)

**Files:**
- Create: `scripts/calibrate.py`
- Modify: `medical_uav/env/config.py` (frozen `map_seed` and `loads` after the gate), `docs/CHANGELOG.md`

**Interfaces:**
- Consumes: `MedicalUAVEnv`, `run_episode`, `edf_policy`, `random_policy`, `generate_map`.
- Produces: `outputs/calibration/calibration.csv`, `outputs/calibration/calibration.md`, printed summary. CLI: `--lambdas` (default `0.03 0.04 0.05 0.06 0.07 0.08 0.10 0.12`), `--seeds` (default dev seeds 100000–100019), `--map-check 200`, `--benchmark`.
- For each λ: EDF (θ = 40) and Random on all seeds → mean ± std of orders/day, on-time rate, served rate, expiry rate, rejections, km per delivered order, charge share, busy share, mean pending pool size, and **hidden on-time-feasible count** (number of decisions where an on-time-feasible pending order was outside the 12 slots). `--map-check N`: success rate and mean tries of `generate_map(s, cfg)` for s in 0..N−1, plus for each seed in 0..19 the largest A*/straight-line distance ratio. `--benchmark`: env-only decisions/s over 20 random-policy episodes at medium load.

- [ ] **Step 1: Write `scripts/calibrate.py`** and run it on 2 seeds to check it works: `python scripts/calibrate.py --seeds 100000 100001 --lambdas 0.07` → table printed, csv written.
- [ ] **Step 2: Freeze the map first** — run `python scripts/calibrate.py --map-check 200 --lambdas` (no λ run) and choose `map_seed` = the first seed in 0..19 whose map has at least one pair with A*/straight-line ratio ≥ 1.15 (no-fly zones visibly matter). Write it into `EnvConfig.map_seed`.
- [ ] **Step 3: Full calibration on the frozen map** `python scripts/calibrate.py --benchmark` (expect several minutes; use `multiprocessing.Pool` over seeds only if it takes > 10 min).
- [ ] **Step 3b: Apply gate G1 (§12)** — choose `loads` so that medium has an EDF on-time rate of 55–85 % and a charging share of 15–35 %, low is clearly easier, and high is overloaded at peaks. If no λ satisfies medium, adjust knobs in the order λ → `charge_rate` → map size. Any change to `charge_rate` or map size invalidates the table: re-run Step 2 (if the map changed) and Step 3, and log each change. Write the chosen `loads` into `EnvConfig`.
- [ ] **Step 4: Run the full test suite** `python -m pytest -q` → all PASS with the frozen values (fix any test that hard-coded the old defaults only if the spec allows that value to change).
- [ ] **Step 5: Record** the calibration table, chosen λ values, `map_seed`, map success rate and throughput in `docs/CHANGELOG.md` (M1 section).
- [ ] **Step 6: Commit** `git commit -m "chore: calibrate load levels and freeze map (gate G1)"`

---

### Task 14: Submission package for 10/16

**Files:**
- Modify: `README.md`, `environment.yml` (only if versions changed)
- Create: `docs/reports/m1/outline.md`, `docs/reports/m1/figures/` (map figure, 3 frames from the EDF demo, calibration table as png), `docs/reports/m1/ai_declaration.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: README** — sections: overview (3 sentences); installation (`conda env create -f environment.yml && conda activate medical-uav && pip install -e .`); quick start (`python scripts/demo.py --policy random` and `--policy edf`); repository structure (the File Structure tree above, one line per file); environment API (spaces, `reset` options, `info` keys, action table); running tests (`python -m pytest -q`); libraries used and why (numpy, gymnasium, matplotlib, imageio; everything else written from scratch); AI-use declaration.
- [ ] **Step 2: Fresh-environment check**

```bash
conda env create -f environment.yml -n medical-uav-check
conda run -n medical-uav-check pip install -e .
conda run -n medical-uav-check python -m pytest -q
conda run -n medical-uav-check python scripts/demo.py --policy random --seed 1 --out outputs/check.gif
conda env remove -n medical-uav-check -y
```
Expected: all tests pass and the gif is written.

- [ ] **Step 3: Report materials** — `outline.md` with the three required sections (Code explanation: module diagram + main loop; Existing codes/libraries; Reflections/Lessons: changes from the proposal, taken from §14 of the spec and `docs/CHANGELOG.md`, plus the calibration outcome). Jihang writes the 2-page text from it; AI helps only with figures and proofreading. `ai_declaration.md` is a draft paragraph naming Claude and Codex and what they were used for.
- [ ] **Step 4: Commit and tag** `git add -A && git commit -m "docs: README and M1 report materials"` then `git tag m1-submission`.
- [ ] **Step 5: Build the zip from the tag** `git archive --format=zip -o outputs/group17_m1.zip m1-submission` and check (`unzip -l`) that it contains `environment.yml`, `README.md`, `medical_uav/`, `scripts/`, `tests/` and the README changes.

---

## Roadmap for later milestones (separate plans, written when each starts)

| plan | written on | main tasks |
|---|---|---|
| M2 Neural network (due 10/30) | 10/17 | FCFS / Nearest / Slack-aware baselines; `evaluate.py` (paired eval, bootstrap CI, CSV/plots); torch added to the env; MLP and set networks (`agents/networks.py`) with empty/single-slot/permutation tests; behaviour cloning on Slack-aware; minimal masked PPO (`agents/ppo.py`) end-to-end on low load; probes P1–P2; end-to-end throughput; gate G2 |
| M3 Learning agent (due 11/6) | 10/31 | set-network PPO; curriculum low → medium → mixed; 3 seeds; probes P3–P5; stall diagnostics in evaluation; dev comparison vs EDF/Slack-aware; set-net drop date 11/4; freeze on 11/6; gate G3 |
| Final (due 11/20) | 11/7 | rolling-horizon planner (drop date 11/11); full paired test evaluation; capacity ablation 1/2/3; optional stretch items; side-by-side video; figures; final zip with trained model |
