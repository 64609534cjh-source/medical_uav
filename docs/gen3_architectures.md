# Gen3: urgency preservation and PICK–DELIVER coordination

Gen2 is frozen at `9136957`. This stage preserves the original 403-feature observation, 26 actions, top-12 candidates, simulator, reward, masked PPO and M5 late learning-rate schedule. The user explicitly retained overall equal-load daily Dev on-time selection, with reward as tie-breaker; no type weights, service tolerances or guardrails are introduced. These are separate architectural comparisons, not capacity-matched proofs of individual mechanisms.

## A: explicit physical-time cargo risk (urgency only)

The reference DELIVER scorer is 235 → 64 Tanh → 1. Add eleven deterministic current-observation features per destination, giving 246 → 64 Tanh → 1. All reference encoders, shared context, PICK, other-action and critic dimensions stay the same. Parameters increase from79,675 to80,379.

Use present cargo rows. The expired flag separates live cargo from expired cargo still physically aboard; all physical cargo remains in the original learned mean/count features. New features are live count/3, live weight/2kg, has-live, minimum live deadline minutes/30, minimum live expiry minutes/30, live due/overdue within15min count/3, expired physical count/3, expired physical weight/2kg, has-other-destination-live, and minimum other-destination live deadline/expiry minutes/30. Empty minima are exactly zero with explicit presence flags; time summaries are clipped to[-300,300] minutes before scaling.

Reconstruct minutes **before** minima: pending/cargo remaining deadline and expiry use the order type’s on-time limit, not a common minute unit. The fixed type limits are30/90/60/75 minutes. For example, medicine9/90=.1 must not outrank plasma6/30=.2 as more urgent: the physical minimum is6 minutes. The new network reads type one-hots and existing normalized fields only; original observation clipping cannot be reversed. The15min due threshold is half the shortest original30min window, frozen before training rather than fitted to any Test result. Overdue but not yet expired live cargo is included in the due count.

## B: shared destination context for PICK (coordination only)

Keep the reference DELIVER head unchanged. For each facility, reuse its107-dimensional local group vector: cargo mean32 + visible pending mean64 + cargo count/3 + pending count/12 + current observed travel/10 + facility identity embedding8. PICK receives context128, order embedding64 and the candidate destination’s same107-dimensional group vector:299 → 64 Tanh → 1. Parameter count86,523. No urgency features are added in this variant.

The visible pending group **includes the candidate itself**, both in its learned mean and count. This is a current-state summary, not an imagined post-pick state. The candidate’s weight/count is never counterfactually added again. Padding uses presence, independently of legal actions; a real temporarily unpickable order participates. No hidden order details, new pairwise distances or future arrivals are supplied.

Both variants retain pending-slot PICK equivariance and cargo-slot invariance. Tests exercise empty/noisy padding, physical-minute conversion across types, expired cargo, self-inclusion, isolated head changes, actual PPO backward passes, checkpoint reload, optional inference equivalence and forced tie fallback. No combined model or expanded candidate interface is part of this stage.

## Diagnostics and uncertainty

The pre-training diagnosis replays frozen M5 and Gen2 destination models on all100 Dev-extra demand days × three loads × five learners. Removing the added failure instrumentation reproduces every prior complete-day field and diagnostic exactly. Four type outcomes (on-time, late, expired-unpicked and expired-after-pick) account for every released order. Their paired daily-rate differences sum to zero.

Failure logs preserve first-visible direct slack, never-visible failures, pre-pick direct slack, deadline-crossing action kinds and urgent-cargo order-state counts. Full simulator details are read only by the observer after policy action selection; they never become network inputs. A deadline crossing during CHARGE or PICK does not prove that action was avoidable. Direct-time checks ignore capacity/energy and alternative routes. Repeated order-state observations are not independent decisions or demand days.

Primary comparisons use paired crossed learner-ID/day-ID product bootstrap,10,000 repetitions, shared day weights across loads, equal-load daily on-time rates and five-learner sample SD. All four types and loads have eligible day counts, release counts and pointwise intervals. Type bootstrap renormalizes each resample’s day weights over zero-release-day exclusions, verifying common demand-only missingness. These intervals are exploratory and have no multiplicity correction. Five learners and one map limit generalization.

## Equivalent inference

Gen2 already groups/scorers all six facilities in batched tensor operations. There is no six-call Python loop to remove. The optional `DestinationInferencePolicy` reuses context projections in both PICK and DELIVER:

`W[c,h] + b = Wc*c + Wh*h + b`.

The underlying original model owns unchanged parameter tensors; no optimized checkpoint or training architecture is registered. Near legal-logit ties≤1e-5 fall back to canonical categorical argmax with the original batch shape. The original network remains the collector/PPO/scientific Test evaluator. Full Dev trajectory checks compare logits, values, actions and every saved day field, including best/last reference models. Existing generic argmax-only inference is also validated on those days. Floating-point logits are not claimed bitwise identical for the factored path.

CPU latency is measured after project training/evaluation jobs stop: same archived real inputs, one thread, batches1/8, inference_mode, three order-rotated rounds,200 warmups and2,000 timed calls per variant/model/batch. Quantiles retain all rounds; aggregate medians are not pooled p95s. The fastest path is measured, not assumed from MAC reduction. [PyTorch Linear](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Linear.html), [benchmark guidance](https://docs.pytorch.org/docs/2.10/benchmark_utils.html).

## Freeze and reproduction

`configs/gen3_holdout_freeze.json` records all prior file hashes, unchanged environment/PPO/collector sources, legacy network syntax hashes, five reference models, and the pre-development unused-local-record audit for Test-v3 IDs400000–400099. Original Test and Test-v2 remain consumed and are not rerun. Ten scratch runs use learner seeds1–5 per new architecture,1,024,000 transitions and1,000 updates each;11 full Dev-main checkpoints per run. Every method/learner is retained for final comparison regardless of Dev ranking. Model/statistical hashes are sealed again before one final Test-v3 evaluation.

Commands require the installed project Python environment, project-root working directory and retained baseline outputs (ignored by Git, preserved locally). Output destinations must be fresh; recorded evidence is never overwritten.

```bash
python scripts/run_gen3.py --output-dir outputs/gen3_architectures
python scripts/evaluate_suite.py --protocol configs/gen3_baseline_diagnostics_protocol.json --output-dir outputs/gen3_baseline_diagnostics
python scripts/analyze_gen3_initial.py
python scripts/analyze_gen3.py --stage dev
```

For one real forward/backward/checkpoint demonstration use `scripts/train.py --architecture urgency` or `--architecture coordinated`, `--total-steps 1024 --eval-split train`, and a fresh `--output-dir`. Full selected-model inference verification and timing protocols preserve exact model paths. Test-v3 is final confirmation for this generation, never a repeated screen for future designs. No theoretical optimum, new-map guarantee or causal error decomposition is claimed.

Codex assisted architecture implementation, tests, training orchestration, diagnostics, uncertainty analysis, inference verification and reporting under the supplied study protocol. The submitting authors remain responsible for checking/explaining the code and results and disclosing AI use in coursework.

The guarded inference adapter is optional:

```python
from medical_uav.agents.destination_inference import DestinationInferencePolicy
# `model` is an original destination/urgency/coordination network loaded normally.
policy = DestinationInferencePolicy(model).eval()
action, _, _ = policy.act(observation_tensor, legal_mask_tensor, deterministic=True)
```

Keep the original network for stochastic collection, PPO probabilities and canonical scientific evaluation. The adapter owns no additional learned weights; the wrapped model retains the ordinary checkpoint format. Benchmark tables distinguish this guarded path from the pre-existing generic argmax-only helper.
