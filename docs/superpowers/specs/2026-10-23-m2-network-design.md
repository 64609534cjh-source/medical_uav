# M2 Neural Network (and the start of M3) — Design

Group 17 · design date 2026-10-23 · M2 due 2026-10-30 23:59 · M3 due 2026-11-06

This document extends the project design (`2026-10-10-medical-uav-design.md`) and is governed by the roadmap `2026-10-23-roadmap-to-final.md` (environment v2, seeds, agents, schedule). Where they differ, the roadmap wins.

**Ownership (owner, 2026-10-23):** the owner hands the technical design of M2 and M3 to Claude and Codex. Work aims straight at M3; the M2 submission is cut from `main` on 10/29 with whatever M2 needs, while M3 work continues on its branch. Choices below marked "decided" are Claude+Codex decisions; the owner is told the results.

## 1. Intent

M2 delivers the policy/value networks that the M3 learning agent will train, a demo that shows them working on real environment data, and a 2-page report. M1 scored 11/15 (class mean 10.5), so M2 must show clearly that the network is correctly built and actually learns something, not only that it runs.

The networks are designed for PPO from the start (masked action logits plus a value), so M3 reuses them unchanged.

## 2. Goals, non-goals, anti-goals

**Goals (M2 submission)**
- Two networks with one interface: an MLP (reference, debugging) and the set/attention network from the proposal (main).
- A demo that creates the networks, runs a forward pass on real observations and masks, shows masked action probabilities and values, runs a backward pass with a real loss and shows gradients and an optimizer step.
- Behaviour cloning (BC) from Slack-aware as the evidence that the networks learn: held-out accuracy and a closed-loop day driven by the trained network.
- Two new baselines promised in the M1 report: FCFS and nearest-first.
- Environment v2 (roadmap §3): compromise reward and the 452-feature observation, with tests; the seed manifest of roadmap §4.
- One shared evaluation script used for every method from now on (shipped in the submission).
- Tests for the network properties below.
- A readable visualization replacing the M1 GIF (roadmap §8), reviewed by the owner before submission.

**Started during M2 week, delivered in M3** (on branch `m3-ppo`, not in the M2 submission): production masked PPO run end to end on both networks (short runs, checkpoint save/reload, measured throughput), and the planner information audit.

**Non-goals for M2**
- Teammates' independent implementations (later).
- Random-map generalization, more architectures, hyper-parameter searches.

**Anti-goals**
- A network that only "runs": M2 must show learning (BC curves and closed-loop results).
- Any change that breaks M1 behaviour or the M1 demo.
- PPO code that is half-finished inside the M2 submission.

## 3. Constraints

- Own PyTorch code; no RL library. Lean, readable, student-level code with short comments; the owner must be able to explain it.
- `environment.yml` and `requirements.txt` pin exact versions, including torch. Installation must work first time on a grader's machine (Linux, and Windows was used by a teammate); CPU must be enough to run the demo in a few minutes. A CPU-only torch wheel is preferred to keep the install small, if it works on Linux, Windows and macOS (builder to verify).
- Git: work on `m2-network`; merge to `main` when it passes tests; tag `m2-submission-v1` on `main` at submission; zip via `git archive` + report PDF. `docs/`, `tests/` and internal scripts stay out of the zip.
- Evaluation seeds from `medical_uav/splits.py` (roadmap §4): training demand seeds for BC data, screen days 100000–100039 for all M2 numbers, selection and test days untouched.

## 4. Networks (decided)

Input: the observation `[B, 452]` = global 80 (the M1 31 features plus the 7×7 travel-time matrix), then 12 pending slots × 27, then 3 cargo slots × 16 (layout fixed by `observation.py`; feature 0 of each slot is its "slot holds an order/box" flag). Output for both networks: `logits [B, 26]` with invalid actions masked to −1e9 using `mask [B, 26]`, and `value [B]`.

**(a) MLP:** separate actor and critic, each `452 → 256 → 256 → (26 | 1)`, tanh (the activation of the successful PPO pilot).

**(b) Set/attention network (main):**

| part | shape |
|---|---|
| pending encoder, shared over 12 slots | `27 → 64 → 64`, ReLU |
| cargo encoder, shared over 3 slots | `16 → 64`, ReLU |
| global encoder | `80 → 64`, ReLU |
| attention | one 4-head cross-attention, width 64; the global embedding queries the pending and cargo embeddings plus one learned always-valid null token; empty slots are masked out of attention |
| context | `[global embedding, attention output]` = 128 |
| PICK(k) logits | one shared scorer on `[pending embedding k, context]`: `192 → 64 → 1` |
| other 14 logits (DELIVER ×6, CHARGE ×6, RETURN, WAIT) | `128 → 64 → 14` |
| value | `[context, masked mean of pending embeddings]`: `192 → 64 → 1` (zero mean when no orders) |

No positional encoding, dropout or extra transformer blocks. Attention masks only empty slots; infeasible-but-present orders stay visible (the action mask handles feasibility). Action index layout must follow `actions.py`.

**Required properties (tests):** masked actions get probability 0; outputs finite for an empty pool, a single order and a low battery; permuting pending slots permutes the PICK logits and leaves the other logits and the value unchanged (set network); a tiny batch can be overfit; shapes and parameter counts as stated.

## 5. Behaviour cloning (decided)

- Teacher: Slack-aware. Data: 900 episodes on training demand seeds, balanced over the three loads; store `(obs, mask, teacher action, demand seed, load)` per decision.
- Split 80/20 by demand seed, a seed's episodes at all loads on the same side.
- Loss: cross-entropy of the masked policy on the teacher action (the value head is not trained by BC; PPO trains it). Adam 3e-4, batch 256, at most 30 epochs, best validation loss kept. Same recipe for both networks.
- Report: train/validation loss curves; validation accuracy overall and per action family (PICK / DELIVER / CHARGE / RETURN / WAIT); closed-loop on-time rate on the 40 dev days per load for each network vs Slack-aware, Random, EDF, FCFS and nearest-first.
- The trained checkpoints (small files) ship with the M2 submission so the demo needs no training.
- M3, not M2: BC from the audited visible-order planner, and BC → PPO from both teachers, as controls (roadmap §5).

## 6. Baselines and evaluation (decided)

- **FCFS** and **nearest-first** in `heuristics.py`, with the shared charging rule, exactly as specified in the project design §9.
- **Policy interface for evaluation:** a policy is called with the environment, the current observation and mask, and returns a valid action index. Heuristics may read the environment; networks may only use the observation and mask.
- **`scripts/evaluate.py`** (shared, shipped): runs any set of policies on the same (load, demand seed) days; writes one CSV row per day (method, load, seed, on-time count and rate, served, expired, per-type on-time, return, charging share); prints a per-load table with mean and a paired 95 % bootstrap CI of the difference against Slack-aware. Asserts that no invalid action is ever chosen.

## 7. Reward (decided, roadmap §3)

Compromise reward: +1 per on-time delivery, +0.1 per late delivery, −0.1 per expired order, −0.001 per battery % used. Chosen from the planner screen and the PPO comparison (roadmap §9.1, §9.4). It is an environment change after M1 and is stated in the M2 report's reflections. The M1 reward and on-time-only stay available for the M3 ablations.

## 8. M2 demo (decided)

`scripts/demo_network.py` (in the submission, run in a few minutes on CPU):
1. builds both networks and prints their parameter counts;
2. collects a small batch of real observations and masks from the environment and prints the shapes;
3. forward pass: prints logits and values, and for a few states the top three valid actions with probabilities in plain words (e.g. "PICK slot 2 (plasma, M→H3) 0.61");
4. backward pass: computes the BC loss on that batch, prints the loss and gradient norms per part, takes one optimizer step and shows the loss decrease;
5. loads the shipped BC checkpoints and runs one full dev day with each network, printing the same day summary as `demo.py`.

`scripts/demo.py` (M1) keeps working; it may gain `--policy mlp|set` to drive the M1 replay with a trained network.

## 9. Schedule

See roadmap §7 (10/23–10/30).

## 10. M2 report outline

Page 1: task and interface in one paragraph; architecture figure with tensor shapes and parameter counts; masking; BC loss and optimizer; code entry points and demo output. Page 2: one results table (BC accuracy, closed-loop on-time vs baselines) and the learning curve; existing libraries (PyTorch) and AI use; reflections as in roadmap §11 (network changes from the proposal; environment v2 changes and why; simplifications; what changes for M3). Do not repeat the M1 environment description.

## 11. Left to the builder

File and function names; torch version and wheel source (subject to section 3); plot style; how the evaluation CSV and table are formatted; the policy-adapter details for heuristics.
