# Rolling-horizon optimizer: pre-registered design (2026-10-27)

Owner's request. The proposal promised a rolling-horizon optimizer; roadmap §10 (10/23) replaced it with the visible-order planner and the scenario rollout. Written before any result of this method. The locked-test manifest was frozen earlier, so the optimizer is not a locked-test entrant: every comparison below is exploratory development evidence.

**Method** (`medical_uav/baselines/horizon.py`). At every decision the optimizer builds a deterministic planning copy of the orders it knows, simulates every sequence of **3** allowed actions exactly, lets Slack-aware finish all known work after each sequence, and takes the first action of the best sequence (environment reward; ties: Slack-aware's choice, then the lowest index). It re-plans at the next decision. Two information sets:
- *visible*: what the agent observes — time, place, battery, cargo and the up to 12 orders shown (as `planning_root` of the scenario rollout). A conventional comparator with the agent's information.
- *foresight*: all pending orders, including those beyond the 12 slots, plus the real orders released in the next **60** minutes. Information no dispatcher has; an optimistic reference, not a competitor.

Fixed before results: depth 3, horizon 60 minutes, Slack-aware as the completion policy. No tuning. The depth was set by timing alone, on screen day 100000 at high load, where every method (Slack-aware, both planners, PPO and this optimizer) delivers every order on time, so no performance difference was seen: depth 2 took 37 / 105 ms per decision on average (visible / foresight), depth 3 271 / 914 ms (p95 1.1 / 3.1 s), affordable for 160 days; depth 3 is kept.

**Evaluation.** Map 0, the 40 screen days, medium and high load; the same days as the existing rows of Slack-aware, the visible-order rollout, the scenario rollout (16) and the five selected PPO checkpoints (`runs/m3_screen_eval_all.csv`, `runs/scenario_screen.csv`). Mean daily on-time, served and pooled per-type rates; PPO seeds averaged per day; paired bootstrap over days (10 000 resamples, seed 0); per-load reading; decision times reported.

**Readings.**
1. *Is visible a stronger conventional comparator?* Visible − visible-order rollout and − scenario rollout, per load.
2. *PPO against it:* five-seed PPO − visible, per load, read against the planner margin of roadmap §1 (−2 points), exploratory.
3. *Headroom:* foresight − five-seed PPO. The search is not exact, so the foresight score is what perfect information of the next hour allows at least, not an upper bound on the optimum. If the difference is below 1 point, this planner gives no evidence of large remaining headroom; if larger, at least that much room exists with perfect information, which does not show that it is reachable with the agent's information.

After the locked test is complete, the visible optimizer may also run on the test days of map 0, reported separately as a post-hoc comparison.
