# Generation 2: two independent architecture hypotheses

Generation 1 is retained at `3c05ad3`. This stage tests the destination and relational changes independently. No new information, action, reward, simulator rule, optimizer setting, training distribution, or training budget is introduced. The top-12 interface stays at 403 observations and 26 actions. Both new methods use five fresh scratch initializations with learner IDs 1–5, as in M5; M5's five frozen models are the reference rather than being retrained.

## Destination-conditioned DELIVER

The global/pending/cargo encoders, masked means, context, PICK scorer and critic retain M5 dimensions. Only the six DELIVER outputs are replaced; the other eight outputs retain a linear context-to-logit map. This is an architectural comparison with scratch learning, not a claim that all shared parameters follow the same optimization trajectory.

Pending destination is read from observation columns 11–16 inside each 27-feature row. Cargo destination is columns 5–10 inside each 16-feature row. Travel times start at global column 16. All column indices here are zero-based. Facility IDs are the six original facility nodes, in the original action order.

For each facility, mean the embeddings of present visible pending orders and physical cargo with that destination. A group with no members returns exactly zero. Unpickable real pending orders participate. Expired cargo still physically aboard participates. No hidden order details enter the network.

The shared DELIVER scorer receives 235 features: context 128, cargo mean 32, pending mean 64, cargo count/3, pending count/12, the existing travel time/10, and a trainable facility embedding of width 8. The scorer is 235 → 64 Tanh → 1. PICK and the critic retain M5 readout structure. Parameter count is 79,675 (M5: 65,232).

## One-layer relational model

Create 16 tokens: global 64, 12 shared pending embeddings of width 64, and three cargo embeddings projected 32 → 64. No candidate-slot positions or extra pairwise geographic distances are supplied. Existing type one-hots remain ordinary input features; no additional type embedding is added.

Use one pre-normalized residual attention block, width 64, four heads of width 16, attention scale 1/√16. A second pre-normalized residual feed-forward block is 64 → 128 Tanh → 64. Dropout is zero. The explicit q/k/v implementation has identical train/eval behavior and no inference-only fast-path switch.

The key mask uses presence, independently of legal actions. The global token always participates, so an entirely empty order/cargo set is well-defined. Mask absent token outputs before pooling. Read out the updated global token, mean updated pending tokens, and cargo tokens projected 64 → 32 before a masked mean. This retains the original 160 → 128 → 128 context encoder and original PICK, other-action and critic head dimensions. Parameter count is 102,896.

Both architectures preserve pending-slot permutation equivariance of PICK logits and invariance of other logits/value, together with cargo-slot permutation invariance. Tests cover masks, empty/noisy padding, expired cargo destination grouping, real-but-unpickable rows, arbitrary leading batch dimensions, real collector/PPO updates and checkpoint reload.

## Frozen protocol and evidence

`configs/gen2_holdout_freeze.json` records first-generation file/model hashes, the local unused-day audit, and new Test-v2 IDs 300000–300099. `configs/gen2_protocol.json` records all ten configurations and frozen source hashes. All runs complete 1,024,000 transitions and 1,000 PPO updates, evaluating full Dev-main at step 0 and every 102,400 transitions. Best uses equal-load mean daily on-time rate, reward as tie-breaker, exact ties retaining the earlier checkpoint. Last is retained independently.

Dev-extra (100040–100139) remains development. Read-only congestion instrumentation uses full simulator orders solely to count hidden possible batching partners and congestion exposure. It does not alter actions, observation construction, rewards or model selection. A destination match or a direct single-order timing check does not prove a feasible multi-order route. Day-difficulty confounding prevents causal interpretation of service correlations.

Both new methods and frozen M5, all five learners each, are predeclared Test-v2 candidates regardless of development ranking. Model hashes and analysis code are sealed again before opening Test-v2. The original Test is not reused for this decision. The statistical unit is learner-ID × matched demand-day-ID; common day resamples span all three loads. Report sample SD and 10,000-repetition paired crossed-bootstrap intervals, including separate learner/day components. Pointwise intervals have no familywise correction, and five learners limit precision. Type rates use daily arithmetic means with zero-release days omitted consistently across methods.

Latency uses the same archived real observations, one CPU thread, canonical masked categorical argmax, batches 1/8, 200 warmups and 2,000 timed calls per model/batch. Compare all 15 best models in isolation and retain absolute microseconds. This comparison uses the same canonical inference interface for all architectures; the separate M5 factored action-only implementation remains available with its historical verified benchmark.

No combined destination+attention network, K=24 expansion, extended budget, reward tuning or additional hyperparameter search is included. This study cannot establish theoretical optimality or generalization to new maps. Small-instance optimal bounds would require a separate study with aligned objectives and simulator constraints.

## References and AI assistance

These are mechanism hypotheses inspired by set modeling, not exact reproductions or promised improvements: [Deep Sets](https://arxiv.org/abs/1703.06114), [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html). Codex assisted in implementing the architectures, tests, reproducible experiment runners, diagnostics, statistical analysis and reporting under the supplied protocol. The submitting authors remain responsible for checking and explaining the code, results and final coursework submission.
