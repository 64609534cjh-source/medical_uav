# M2 Neural Network (and the start of M3) — Design

Group 17 · design date 2026-10-23 · M2 due 2026-10-30 23:59 · M3 due 2026-11-06

This document extends the project design (`2026-10-10-medical-uav-design.md`). Where the two differ, this one wins for M2.

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
- One shared evaluation script used for every method from now on.
- Tests for the network properties below.

**Started during M2 week, delivered in M3** (on branch `m3-ppo`, not in the M2 submission): masked PPO, the reward screen (section 7), and the rollout-planner baseline.

**Non-goals for M2**
- Fixing the GIF / visualization (TA comment "too fast to understand anything"). Recorded; the owner will discuss it later.
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
- Evaluation seeds from `medical_uav/splits.py`: training demand seeds for BC data, dev seeds (100000–100039) for all M2 numbers, test seeds untouched.

## 4. Networks (decided)

Input: the observation `[B, 403]` = global 31, then 12 pending slots × 27, then 3 cargo slots × 16 (layout fixed by `observation.py`; feature 0 of each slot is its "slot holds an order/box" flag). Output for both networks: `logits [B, 26]` with invalid actions masked to −1e9 using `mask [B, 26]`, and `value [B]`.

**(a) MLP:** separate actor and critic, each `403 → 256 → 256 → (26 | 1)`, ReLU.

**(b) Set/attention network (main):**

| part | shape |
|---|---|
| pending encoder, shared over 12 slots | `27 → 64 → 64`, ReLU |
| cargo encoder, shared over 3 slots | `16 → 64`, ReLU |
| global encoder | `31 → 64`, ReLU |
| attention | one 4-head cross-attention, width 64; the global embedding queries the pending and cargo embeddings plus one learned always-valid null token; empty slots are masked out of attention |
| context | `[global embedding, attention output]` = 128 |
| PICK(k) logits | one shared scorer on `[pending embedding k, context]`: `192 → 64 → 1` |
| other 14 logits (DELIVER ×6, CHARGE ×6, RETURN, WAIT) | `128 → 64 → 14` |
| value | `[context, masked mean of pending embeddings]`: `192 → 64 → 1` (zero mean when no orders) |

No positional encoding, dropout or extra transformer blocks. Attention masks only empty slots; infeasible-but-present orders stay visible (the action mask handles feasibility). Action index layout must follow `actions.py`.

**Required properties (tests):** masked actions get probability 0; outputs finite for an empty pool, a single order and a low battery; permuting pending slots permutes the PICK logits and leaves the other logits and the value unchanged (set network); a tiny batch can be overfit; shapes and parameter counts as stated.

## 5. Behaviour cloning (decided)

- Teacher: Slack-aware. Data: about 1,000 episodes on training demand seeds, balanced over the three loads; store `(obs, mask, teacher action)` per decision.
- Split by whole episodes, 80/20 train/validation.
- Loss: cross-entropy of the masked policy on the teacher action (the value head is not trained by BC; PPO trains it). Adam 3e-4, batch 256, at most 20 epochs. Same recipe for both networks.
- Report: train/validation loss curves; validation accuracy overall and per action family (PICK / DELIVER / CHARGE / RETURN / WAIT); closed-loop on-time rate on the 40 dev days per load for each network vs Slack-aware, Random, EDF, FCFS and nearest-first.
- The trained checkpoints (small files) ship with the M2 submission so the demo needs no training.
- M3 option, not M2: use the rollout planner (stronger than Slack-aware) as the teacher, so BC → PPO starts above Slack-aware. Evaluated in the M3 design.

## 6. Baselines and evaluation (decided)

- **FCFS** and **nearest-first** in `heuristics.py`, with the shared charging rule, exactly as specified in the project design §9.
- **Policy interface for evaluation:** a policy is called with the environment, the current observation and mask, and returns a valid action index. Heuristics may read the environment; networks may only use the observation and mask.
- **`scripts/evaluate.py`** (shared, internal): runs any set of policies on the same (load, demand seed) days; writes one CSV row per day (method, load, seed, on-time count and rate, served, expired, per-type on-time, return, charging share); prints a per-load table with mean and a paired 95 % bootstrap CI of the difference against Slack-aware. Asserts that no invalid action is ever chosen.

## 7. Reward screen (started in M2 week, owner decides)

The 2026-10-15 review found that the current reward mostly buys rescues of late orders rather than on-time deliveries. Before any long PPO run, compare three candidates with the rollout planner (visible orders only) on the 40 dev days at medium and high load:

1. current reward;
2. on-time only: +1 per on-time delivery, nothing else;
3. compromise: +1 on time, +0.1 late, −0.1 expiry, current energy cost.

Report on-time, served and expiry rates and per-type on-time for each. Claude and Codex choose by 2026-10-26 from these numbers (prefer the candidate with the highest on-time rate that does not starve any order type), tell the owner, and freeze the choice for M3. A changed reward is an environment change after M1 and must be stated in the M2 report's reflections.

## 8. M2 demo (decided)

`scripts/demo_network.py` (in the submission, run in a few minutes on CPU):
1. builds both networks and prints their parameter counts;
2. collects a small batch of real observations and masks from the environment and prints the shapes;
3. forward pass: prints logits and values, and for a few states the top three valid actions with probabilities in plain words (e.g. "PICK slot 2 (plasma, M→H3) 0.61");
4. backward pass: computes the BC loss on that batch, prints the loss and gradient norms per part, takes one optimizer step and shows the loss decrease;
5. loads the shipped BC checkpoints and runs one full dev day with each network, printing the same day summary as `demo.py`.

`scripts/demo.py` (M1) keeps working; it may gain `--policy mlp|set` to drive the M1 replay with a trained network.

## 9. Schedule

| date | work |
|---|---|
| 10/23–24 | torch in environment files; both networks + tests; FCFS, nearest-first; `evaluate.py` |
| 10/25–26 | BC data and training; `demo_network.py`; reward screen data to the owner |
| 10/27–28 | closed-loop evaluation table; README for M2; report draft; PPO starts on `m3-ppo` |
| 10/29–30 | freeze; fresh-environment and extracted-zip check; tag `m2-submission-v1`; submit |
| 10/31–11/6 | M3 (own design and plan once its rubric is known): PPO pilot, reward frozen, rollout baseline, training runs, comparison |

## 10. M2 report outline

Page 1: task and interface in one paragraph; architecture figure with tensor shapes and parameter counts; masking; BC loss and optimizer; code entry points and demo output. Page 2: one results table (BC accuracy, closed-loop on-time vs baselines) and the learning curve; existing libraries (PyTorch) and AI use; reflections: changes from the proposal's network (separate encoders, fixed heads for non-PICK actions, top-12 visibility, MLP kept as reference), simplifications, and what changes for M3 (PPO, reward). Do not repeat the M1 environment description.

## 11. Left to the builder

File and function names; torch version and wheel source (subject to section 3); the exact BC data size within "about 1,000 episodes"; plot style; how the evaluation CSV and table are formatted; the policy-adapter details for heuristics.
