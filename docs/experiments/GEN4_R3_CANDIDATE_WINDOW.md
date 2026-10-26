# GEN4-R3-CANDIDATE-WINDOW — staged registration

Status: interface engineering and control checks authorized; learning experiment not yet launched. Owner: Codex implementation; human reviewer pending. Category **B: information/action interface**, not a pure architecture claim. Baseline is the frozen destination policy and original physical simulator/reward.

Hypothesis: extra selectable/visible pending orders can improve service when K=12 misses timely opportunities. R1 prevalence and timing results determine the investment; overflow by itself is not evidence of recoverable loss. Primary factor: K=12 versus K=24 with the original deterministic deadline/expiry ranking. Do not mix a new ranking or new reward into this comparison.

| Contract | K=12 control | K=24 candidate |
|---|---|---|
| Observation | 31 + 12×27 + 3×16 = 403 | 31 + 24×27 + 3×16 = 727 |
| Actions | 12 PICK + 14 = 26 | 24 PICK + 14 = 38 |
| Physics/reward/termination | frozen | identical |
| Encoders/head architecture | destination shared order scorers | same shared scorers, no added learned slots |
| Count normalization | original physical scales | retain original scales; no incidental rescaling |

Observation rows, mask and PICK index must refer to the same order. DELIVER/CHARGE/RETURN/WAIT keep their meanings and shift by K−12. Real but unpickable orders remain in the state; absent rows stay masked. Actor/critic only see the versioned observation. Hidden simulator fields are permitted solely in diagnostics. Candidate generation itself uses the same public-to-interface selection rule as the frozen task.

Engineering acceptance before training: K=12 reproduces full canonical trajectories, observations, masks, rewards, all day fields and rule choices; K=24 extra slots actually map to different orders; all three rules use the same K-specific interface; capacities/energy/expiry hold. Generalized networks must agree with old K=12 weights/forward, support padding/permutation, and pass actual masked forward/backward/checkpoint reload. No monkeypatching global constants or mutating frozen modules.

Planned bounded first screen: A0 and A1 × learners 11/12 × 262,144 real transitions (total 1,048,576), scratch weights, eight envs × 128 decisions, unchanged PPO/gamma=1/lambda=.95. Late-linear LR uses the **262,144** budget in both methods, not an inherited 1M schedule. Evaluate step 0 and every 65,536: five equally spaced checkpoints on the same 100×3 `dev_tune` manifest. Select equal-load mean daily on-time, reward tie-break, earliest exact tie; one checkpoint across loads; retain last/all checkpoints and every learner.

This is a new development protocol and cannot be presented as an identical-selection comparison against historical 40-day/11-checkpoint experiments. Full long training, if advanced, restarts with its separately frozen 1,024,000 budget/schedule and long-screen learners. Keep slower-learning mechanism candidates eligible for one long-screen seat.

No `dev_confirm` evaluation before configuration/checkpoint selection is frozen. All confirmation access must be logged; no iteration on its scores. Old Tests stay consumed and unused for selection; Test-v4 remains absent. New manifests/learner reservations and local collision-audit scope are in `configs/gen4_scenario_registry.json`.

Report all loads/types, reward, mutual failure outcomes, visibility/waiting, seed-wise curves, best/last, train versus evaluation cost and inference p50/p95. Matched real transitions are not matched total compute. The plan's ~0.5pp practical promotion threshold remains a proposal to freeze in the actual training protocol; interface correctness alone does not assert a service gain. Stop on illegal actions, nonfinite values, schema mismatch or failed K=12 compatibility; preserve negative runs.
