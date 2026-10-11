# M1 Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every issue found in the M1 review and the Claude–Codex stock-take (rounds 9–11) before the 10/16 report: repository hygiene, configuration validation, simulator cleanup, the Slack-aware heuristic, recalibrated loads, a clearer demo and verified installation.

**Architecture:** No change to the environment's dynamics, actions, observation layout or reward. Changes are validation, refactors behind unchanged interfaces, one new policy in `baselines/heuristics.py`, a seed manifest, calibration output, demo/render presentation, and packaging rules.

**Tech Stack:** Python 3.11 (conda env `medical-uav-rl`), numpy, gymnasium, matplotlib, imageio, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-medical-uav-design.md` (v2, §12 "M1 remaining" items 1–9).

## Global Constraints

- Comments: concise, programmer-style; no § references and no decorative separators in code.
- Stage explicit paths only (`git add <files>`), check `git diff --cached --stat` before each commit; never `git add -A`.
- Every behaviour change starts with a failing test.
- Loads after this plan: low 0.04, medium 0.08, high 0.12; diagnostic 0.10; stress 0.14 (appendix only).
- Seeds: train [0, 100000); dev 100000–100039; test 200000–200099 (never used in M1).
- Commit trailer as given in the session.

## Review Focus

1. **Load change ripples:** tests, README, calibration.md and report outline must all say 0.04/0.08/0.12 → Task 5 test `test_default_loads` plus a grep step.
2. **export-ignore must not drop needed files:** the archive must still contain code, tests, scripts, README, environment.yml, requirements.txt, pyproject.toml → Task 8 extracted-archive test run.
3. **Validation must not reject the defaults or `check_env`:** Task 2 `test_default_config_is_valid` and the existing `test_check_env`.
4. **Streaming writer:** GIF/MP4 written from a generator must have the right frame count → Task 6 `test_save_animation_streams_generator`.
5. **Slack-aware validity:** only valid actions on every load including overload → Task 4 `test_slack_aware_only_valid_actions` (λ 0.04–0.4).

---

### Task 1: Repository hygiene

**Files:** Modify `.gitignore`, `README.md`; Create `.gitattributes`; untrack `.vscode/settings.json`.

- [ ] **Step 1:** `git rm --cached .vscode/settings.json`; add `.vscode/` to `.gitignore`.
- [ ] **Step 2:** Create `.gitattributes`:
```
docs/superpowers/** export-ignore
docs/reports/** export-ignore
.flake8 export-ignore
```
- [ ] **Step 3:** README: remove the link to the design document (excluded from archives); keep a one-line description of the design instead.
- [ ] **Step 4: Verify** `git archive --format=tar HEAD | tar -t | grep -E "^(docs/|\.vscode)"` → no output after committing.
- [ ] **Step 5: Commit** `chore: ignore editor files, export-ignore design and report drafts`.

---

### Task 2: Configuration validation and naming cleanup

**Files:** Modify `medical_uav/env/config.py`, `observation.py`, `uav_env.py`, `simulator.py`, `masks.py`, `render.py`, `physics.py`, `orders.py`, `map_gen.py`, `astar.py`; Test `tests/test_config.py`.

**Interfaces:**
- Produces: `EnvConfig.validate() -> None` (raises `ValueError`); `day_length(cfg) -> float` = `cfg.release_window + max(t.expiry for t in ORDER_TYPES)` (660 by default); `EnvConfig` without `charge_targets`.
- `MedicalUAVEnv.__init__` calls `cfg.validate()`, raises `ValueError` for an unknown `load` or `render_mode`; `reset` raises `ValueError` for an unknown `load` option or a non-finite/negative `lam`.
- Observation: named scales `DURATION_SCALE = 30.0`, `TRAVEL_SCALE = 10.0`, `COUNT_SCALE = 10.0`, `SAME_DEST_SCALE = 5.0`; time feature uses `day_length(cfg)`; `_pending_features(..., pick_allowed: bool)`.

Validation rules: `grid_size > 0`, `cell_km > 0`, `speed_km_per_min > 0`, `energy_per_km > 0`, `payload_factor >= 0`, `0 <= reserve < 100`, `charge_rate > 0`, `handling_min > 0`, `1 <= max_boxes <= 3`, `max_weight > 0`, `release_window > 0`, `wait_max > 0`, `pool_cap >= 1`, `n_slots == 12`, all loads finite and `>= 0`, `expiry_penalty >= 0`.

- [ ] **Step 1: Failing tests**
```python
# tests/test_config.py
import math
import pytest
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import day_length

def test_default_config_is_valid():
    EnvConfig().validate()
    assert day_length(EnvConfig()) == 660.0

@pytest.mark.parametrize("field,value", [("wait_max", 0.0), ("charge_rate", 0.0), ("max_boxes", 4),
                                         ("n_slots", 16), ("speed_km_per_min", -1.0), ("reserve", 100.0)])
def test_invalid_config_rejected(field, value):
    cfg = EnvConfig(**{field: value})
    with pytest.raises(ValueError):
        cfg.validate()

def test_env_rejects_unknown_load_and_render_mode():
    with pytest.raises(ValueError):
        MedicalUAVEnv(load="extreme")
    with pytest.raises(ValueError):
        MedicalUAVEnv(render_mode="video")
    env = MedicalUAVEnv()
    with pytest.raises(ValueError):
        env.reset(options={"lam": math.nan})
```
- [ ] **Step 2: Run** `python -m pytest tests/test_config.py -q` → FAIL.
- [ ] **Step 3: Implement** validation, `day_length`, named scales, `pick_allowed`; remove `charge_targets`; delete § references and `# ----` separators in all package files (comments stay short).
- [ ] **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `refactor(env): validate config, name scales, drop dead config`.

---

### Task 3: Simulator cleanup and event regressions

**Files:** Modify `medical_uav/env/simulator.py`; Test `tests/test_simulator_core.py`, `tests/test_simulator_actions.py`, `tests/test_env.py`.

**Interfaces:** `Simulator.__init__` raises `ValueError` if orders are not sorted by release; private helpers `_next_release_time()`, `_next_expiry_time()`, `_dispose(order)`; public behaviour unchanged.

- [ ] **Step 1: Failing/regression tests** (line map: M 0, H1 2, H2 3, H3 4, H4 5, H5 6, C2 3.5 km)
```python
def test_unsorted_orders_rejected(line_map):
    orders = [make_order(0, 1, M, 1, 0.5, 10.0), make_order(1, 1, M, 2, 0.5, 5.0)]
    with pytest.raises(ValueError):
        make_sim(line_map, orders)

def test_unload_expiry_release_same_instant_with_full_pool(line_map):
    cfg = EnvConfig(pool_cap=1)
    a = make_order(0, 1, M, 1, 0.5, 0.0)     # medicine to H1
    c = make_order(1, 2, 5, M, 0.3, 3.0)     # sample at H5, expires at 123, fills the pool
    d = make_order(2, 1, M, 2, 0.5, 123.0)   # released exactly when c expires
    sim = make_sim(line_map, [a, c, d], cfg)
    sim.do_pick(a)                            # t = 2
    sim.advance_to(119.0)
    sim.do_deliver(1)                         # arrive 121, unload done 123
    assert [(k, i) for (_, k, i) in sim.events[-3:]] == [("deliver", 0), ("expire", 1), ("release", 2)]
    assert d.status == "pending"

def test_box_expiring_during_other_unload_is_disposed(line_map):
    a = make_order(0, 1, M, 2, 0.5, 0.0)     # medicine, deadline 90
    b = make_order(1, 0, M, 2, 0.6, 61.0)    # plasma, deadline 91, expiry 121
    sim = make_sim(line_map, [a, b])
    sim.do_pick(a)
    sim.advance_to(61.0)
    sim.do_pick(b)
    sim.advance_to(117.0)
    sim.do_deliver(2)                         # arrive 120: a unloaded 120-122, b expired at 121
    assert a.status == "delivered" and a.end_time == pytest.approx(122.0)
    assert b.status == "expired" and b.disposed_time == pytest.approx(124.0)
    assert sim.onboard == []
```
and in `tests/test_env.py`:
```python
def test_same_seed_gives_identical_event_log():
    a, b = MedicalUAVEnv(), MedicalUAVEnv()
    rollout(a, 5)
    rollout(b, 5)
    assert a.sim.events == b.sim.events
```
- [ ] **Step 2: Run** → `test_unsorted_orders_rejected` FAILS; the other three should PASS already (they pin behaviour that the Codex/reviewer probes found correct). If any of them fails, debug before refactoring.
- [ ] **Step 3: Implement** the sorted-order check and the helper refactor.
- [ ] **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `refactor(sim): shared event helpers, dispose helper, sorted-order check`.

---

### Task 4: Slack-aware heuristic

**Files:** Modify `medical_uav/baselines/heuristics.py`; Test `tests/test_heuristics.py`.

**Interfaces:** `slack_aware_policy(env, theta: float = 30.0) -> int`, following spec §9 rules (1)–(5); "can still be on time" for a PICK uses `direct_completion_time`; for a DELIVER(n), `travel_time(here, n) + handling` against the earliest deadline of non-expired boxes for n.

- [ ] **Step 1: Failing tests** (line map fixture; a tiny wrapper gives the policies the env interface)
```python
from medical_uav.baselines.heuristics import edf_policy, run_episode, slack_aware_policy
from medical_uav.env.actions import DELIVER0, PICK0
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import rank_pending
from tests.conftest import make_order, make_sim


class SimEnv:
    """Minimal env-like wrapper around a hand-built simulator."""
    def __init__(self, sim):
        self.sim = sim
        self.slots = rank_pending(sim)

    def action_masks(self):
        return compute_action_mask(self.sim, self.slots)


def test_slack_aware_delivers_here_first(line_map):
    a = make_order(0, 0, M, 2, 0.6, 0.0)
    sim = make_sim(line_map, [a])
    sim.do_pick(a)
    sim.fly_to(2)                      # at H2 with the box for H2 still on board
    sim.battery = 35.0                 # below θ=40: charging would be wanted
    assert slack_aware_policy(SimEnv(sim), theta=40.0) == DELIVER0 + 2


def test_slack_aware_batches_same_destination(line_map):
    a, c = make_order(0, 1, M, 2, 0.5, 0.0), make_order(1, 1, M, 2, 0.5, 0.0)
    sim = make_sim(line_map, [a, c])
    sim.do_pick(a)                     # at M, box for H2 on board, c waits at M for H2
    env = SimEnv(sim)
    assert slack_aware_policy(env) == PICK0 + env.slots.index(c)
    assert edf_policy(env) == DELIVER0 + 2     # EDF delivers first on the deadline tie


def test_slack_aware_prefers_feasible_over_hopeless(line_map):
    late = make_order(0, 0, M, 1, 0.7, 0.0)    # deadline 30: hopeless at t=45
    fresh = make_order(1, 1, M, 2, 0.5, 40.0)  # deadline 130
    sim = make_sim(line_map, [late, fresh])
    sim.advance_to(45.0)
    env = SimEnv(sim)
    assert slack_aware_policy(env) == PICK0 + env.slots.index(fresh)
    assert edf_policy(env) == PICK0 + env.slots.index(late)


@pytest.mark.parametrize("lam", [0.04, 0.08, 0.12, 0.4])
def test_slack_aware_only_valid_actions(lam):
    env = MedicalUAVEnv()
    obs, info = env.reset(options={"demand_seed": 1, "lam": lam})
    done = False
    while not done:
        a = slack_aware_policy(env)
        assert info["action_mask"][a]
        obs, r, done, _, info = env.step(a)


def test_slack_aware_beats_edf_on_medium():
    env = MedicalUAVEnv()
    opts = {"lam": 0.08}
    slack = np.mean([run_episode(env, slack_aware_policy, s, opts)["on_time_rate"] for s in range(10)])
    edf = np.mean([run_episode(env, edf_policy, s, opts)["on_time_rate"] for s in range(10)])
    assert slack > edf + 0.1
```
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(baselines): Slack-aware heuristic`.

---

### Task 5: Metrics, seed manifest, calibration and loads (amended gate G1)

**Files:** Modify `medical_uav/metrics.py`, `medical_uav/env/config.py`, `scripts/calibrate.py`, `tests/test_env.py`, `docs/reports/m1/calibration.md`, `docs/reports/m1/outline.md`; Create `medical_uav/splits.py`.

**Interfaces:**
- `splits.py`: `TRAIN_SEED_RANGE = (0, 100_000)`, `DEV_SEEDS = range(100_000, 100_040)`, `TEST_SEEDS = range(200_000, 200_100)`, `GENERALIZATION_MAPS = range(300_000, 300_020)`, `GENERALIZATION_SEEDS = range(400_000, 400_010)`.
- `episode_metrics` adds `delivered` (int) and `on_time_by_type` (dict name → rate, NaN if no orders of that type).
- `calibrate.py`: policies `slack`, `edf`, `random`; `--lambdas` default `0.04 0.06 0.08 0.10 0.12 0.14`; `--tune-theta` evaluates Slack-aware θ ∈ {20, 30, 40} at medium; output columns add `hidden_pct` (% of decisions with a hidden on-time-feasible order), `hidden_mean`, `pool_p95`, `pool_max`, `rejected`, with the number of days per row.

- [ ] **Step 1: Failing tests** (and change `test_demand_seed_and_load_options` to expect high = 0.12)
```python
def test_default_loads():
    assert EnvConfig().loads == {"low": 0.04, "medium": 0.08, "high": 0.12}


def test_metrics_by_type_and_delivered():
    m = run_episode(MedicalUAVEnv(), slack_aware_policy, 0)
    assert m["delivered"] == round(m["served_rate"] * m["n_orders"])
    assert set(m["on_time_by_type"]) == {"plasma", "medicine", "sample", "supplies"}


def test_splits_disjoint():
    from medical_uav import splits
    lo, hi = splits.TRAIN_SEED_RANGE
    for seed in list(splits.DEV_SEEDS) + list(splits.TEST_SEEDS):
        assert not lo <= seed < hi
    assert not set(splits.DEV_SEEDS) & set(splits.TEST_SEEDS)
```
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**, set the loads.
- [ ] **Step 4: Run calibration:** `python scripts/calibrate.py --tune-theta --map-check 200 --benchmark`. Check amended G1: Slack-aware medium on-time ≈ 0.70–0.85, high clearly lower, low ≥ 0.95; set θ to the best dev value. If G1 fails, stop and report the numbers instead of retuning silently.
- [ ] **Step 5:** Write the table and the G1 decision into `docs/reports/m1/calibration.md`; update `outline.md` load numbers; `grep -rn "0\.06\|0\.10" README.md docs/reports` and fix stale load statements.
- [ ] **Step 6: Run** full suite → PASS. **Commit** `feat: seed manifest, per-type metrics, loads calibrated on Slack-aware`.

---

### Task 6: Demo and rendering presentation

**Files:** Modify `medical_uav/env/render.py`, `medical_uav/env/uav_env.py`, `scripts/demo.py`; Test `tests/test_render.py`, `tests/test_demo.py`.

**Interfaces:**
- `save_animation(frames: Iterable[np.ndarray], path: str, fps: int) -> int` streams frames to an imageio writer and returns the frame count.
- `MedicalUAVEnv.decision_log: list[tuple[float, str, float]]` — (decision time, action name, step reward), reset on `reset`.
- `Renderer.draw(sim, t, header: str = "")`: header line across the top; order-type colour legend; pending list titled "pending (by deadline)" with "+N more" when more than 12.
- `scripts/demo.py`: `--policy {random,edf,slack}` (default `slack`); argument validation (`--fps > 0`, `--minutes-per-frame > 0`, otherwise `parser.error`); every saved/live frame header = `policy | load | demand seed | t | last action`; `--live` falls back when the backend is non-interactive (`matplotlib.get_backend()` in agg/pdf/svg/ps/cairo/template) and prints `partial run` if the window is closed early.

- [ ] **Step 1: Failing tests**
```python
def test_save_animation_streams_generator(tmp_path):
    frames = (np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(5))
    assert save_animation(frames, str(tmp_path / "a.gif"), fps=5) == 5
    assert len(imageio.mimread(tmp_path / "a.gif")) == 5

def test_decision_log_records_actions():
    env = MedicalUAVEnv()
    env.reset(seed=0)
    env.step(25)
    assert env.decision_log[0][1] == "WAIT" and env.decision_log[0][0] == 0.0

def test_demo_rejects_bad_arguments():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/demo.py", "--fps", "0", "--no-video"],
                            capture_output=True, cwd=repo)
    assert result.returncode != 0
```
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS; regenerate the report snapshots with `--policy slack` and look at one frame.
- [ ] **Step 5: Commit** `feat(demo): labelled frames, legend, streamed export, argument checks`.

---

### Task 7: Installation files and README

**Files:** Modify `README.md`, `environment.yml` (only if pins change); Test `tests/test_setup.py`.

- [ ] **Step 1: Failing test** `test_requirements_match_environment_yml`: parse `requirements.txt` and the pip section of `environment.yml`; the pinned sets must be identical.
- [ ] **Step 2: Run** → PASS if already identical (it pins current state; keep it as a guard).
- [ ] **Step 3: README:** primary install = `conda create -n medical-uav-rl --override-channels -c conda-forge python=3.11 pip` + `pip install -r requirements.txt` + `pip install -e .`; `environment.yml` as the alternative; loads 0.04/0.08/0.12; Slack-aware in the baselines list; demo defaults; seeds (`demand_seed` vs `seed`).
- [ ] **Step 4: Commit** `docs: README install path, loads, baselines`.

---

### Task 8: Verification and hand-off

- [ ] **Step 1:** `python -m pytest -q` → all pass.
- [ ] **Step 2:** Fresh environment via the README's primary path in a temporary env → tests pass, `scripts/demo.py --policy slack --seed 1` writes a gif; remove the temporary env (use `conda remove --override-channels -c conda-forge`).
- [ ] **Step 3:** `git archive --format=zip -o outputs/m1_check.zip HEAD`; unzip into a temp dir; run `python -m pytest -q` there with the dev env → pass; confirm no `docs/` or `.vscode/`.
- [ ] **Step 4:** Codex review of the diff of this plan (after its usage limit resets) and a fresh reviewer; fix Critical/Important with TDD.
- [ ] **Step 5:** Hand-off to the student: code walkthrough, real-screen `--live` check, report writing. After the student's review: fast-forward `main`; at actual submission create the annotated tag `m1-submission-v1` and build the zip from it (plus the report PDF).
