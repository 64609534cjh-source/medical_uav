**Changes from the proposal**

- **Shared scorer for PICK only.** The proposal scored every action with one shared scorer; we keep it only for PICK, since only a pickup belongs to a particular order, and score the other 14 actions with a small separate head.
- **Separate order and cargo encoders.** The proposal shared one, but their features differ (27 against 16 per slot).
- **Added an MLP control and imitation learning** to test the design and the pipeline before PPO.

**Simplifications** (the environment's own are listed in the M1 report)

- **The network sees at most 12 waiting orders**, which cost the rollout planner at most 0.5 on-time points.
- **One fixed map** instead of the proposed random maps, so the networks are not yet shown to work on other maps.
- **One training seed per network**: our paired intervals cover variation across days, not across training runs.

**Lessons learned**

- **Decision accuracy is not day performance.** At 94% agreement the MLP still delivered fewer orders on time than Slack-aware, as each remaining mistake can lead to situations the demonstrations never covered; the networks imitating the rollout planner agreed on under 80% of decisions, yet delivered more on time. Agreement measures copying, not performance, so we judge policies on whole unseen days.
- **Look at agreement by decision type.** Imitating the rollout planner, the set network matches 90% of deliveries and 98% of waits but only 70% of pickups and 45% of charges (imitating Slack-aware: all over 93%). The low total comes from which order to take and when to charge: an overall accuracy can hide where two policies differ.
- **Test a reward before training with it.** Scoring the rollout planner with the M1 reward (up to +1 per late delivery) showed that it chased late orders over others; with the new reward it delivers more orders on time at medium load (83.1% → 89.7%) but fewer in total (96.6% → 94.6% served), so we report both rates.
- **Choose the epoch by validation loss, not a fixed budget.** After 30 epochs the set network barely beat the MLP (79.5% vs 79.3% on time), yet it kept improving until epoch 95 and matched Slack-aware, while the MLP stopped improving at epoch 42 and, when imitating the rollout planner, overfitted after epoch 29 (Figure 2).
- **Compare policies on the same days.** Slack-aware's on-time rate ranges from 57% to 96% across medium-load days, so comparing two policies on different days leaves a 95% interval of about ±4 points, more than most gaps in Table 1; on the same days the paired interval shrinks to ±1–2 points.

**Planned changes**

- **PPO:** train both networks with our masked PPO, from scratch and from imitation, five seeds each.
- **Comparison:** against each other, all baselines and stronger planners, on test days never used before.
- **Random maps:** train a version on many random maps, as proposed, and test it on maps it has never seen.
