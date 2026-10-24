# SDD ledger — plan: docs/superpowers/plans/2026-10-23-m2-network.md

Executor: inline (executing-plans), branch m2-network; spec docs/superpowers/specs/2026-10-23-roadmap-to-final.md + 2026-10-23-m2-network-design.md.

Pre-flight (shared interfaces):
- T1 → T2/T3/T4: GLOBAL_DIM 80, OBS_DIM 452 consumed via observation.py constants — consistent.
- T1 → T6/T7/T12: SCREEN_SEEDS/SELECT_SEEDS/TEST_SEEDS/TRAINING_SEEDS — consistent.
- T3/T4 → T5/T7/T11: NETWORKS dict, forward(obs, mask) -> (logits, value) — consistent.
- T5 → T6/T8/T9/T12: save/load_checkpoint, NetworkPolicy(env, obs, mask), probabilities — consistent.
- T6 → T12: T12 consumes `evaluate.run_day` but `scripts/evaluate.py` is not importable from package code. Ruling: the evaluation functions (heuristic, run_day, evaluate, paired_ci) live in `medical_uav/evaluation.py`; `scripts/evaluate.py` is the CLI; tests import `medical_uav.evaluation` — scripts are not a package — cost if wrong: one module move.
- T7 → T8/T9: checkpoints/bc_mlp.pt, bc_set.pt — consistent.
- Task 1: Ruling: delivery_reward drops its unused expiry_after parameter (signature (age, limit, cfg)) — v2 reward no longer depends on expiry time; callers updated — cost if wrong: re-add one argument
- Task 1: Ruling: test_expiry_and_energy_cost_v2 lives in tests/test_simulator_core.py (needs make_sim), not test_physics.py — cost if wrong: none
- Task 1: note: README reward/observation lines are updated in Task 10 (README for M2)
Task 1: complete (commits 618087a..534ec12, tests: python -m pytest -q → 152 passed in 13.56s)
Task 2: complete (commits 534ec12..8ab67ae, tests: python -m pytest -q → 154 passed in 14.29s)
Task 3: complete (commits 8ab67ae..958e76e, tests: python -m pytest -q → 157 passed in 15.09s)
Task 4: complete (commits 958e76e..01e84b7, tests: python -m pytest -q → 161 passed in 15.82s)
Task 5: complete (commits 01e84b7..55c76d0, tests: python -m pytest -q → 164 passed in 15.95s)
- Task 5: note: branch m3-ppo created at 55c76d0 for Part C
- Task 6: Ruling: Random in evaluate.py is re-seeded per day with the demand seed (class DailyRandom), as in M1 calibrate.py — reproduces M1 Table 1 (55.5/26.3/22.6) — cost if wrong: none
- Task 6: result (screen days, env v2): FCFS .970/.493/.321, nearest .992/.716/.532 on-time (low/med/high); Random/EDF/Slack identical to M1
Task 6: complete (commits 55c76d0..432e6b2, tests: python -m pytest -q → 169 passed in 16.33s)
- Task 7: finding: checkpoint info with numpy str keys failed torch weights_only load — fixed (str keys) with test_trained_checkpoint_with_history_loads RED→GREEN
Task 7: complete (commits 432e6b2..688416b, tests: python -m pytest -q → 174 passed in 17.23s)
- Task 8: Ruling: forward-pass examples show the 3 batch states with the most valid actions (not fixed indices) and the demo adds Slack-aware's result on the same day — clearer demo — cost if wrong: none
Task 8: complete (commits 688416b..d18ca18, tests: python -m pytest -q → 175 passed in 18.91s)
- Task 9: Ruling: no separate save_video — save_animation already writes .mp4 — cost if wrong: none
- Task 9: Ruling: explanation_times takes t_end (end of the day) so the last decision has an end time; demo finds a frame's decision by parsing "Decision n" from its caption — cost if wrong: small refactor
- Task 9: Ruling: drone_state_at/draw gain at_end, and event windows exclude events exactly at t0 (found in frame review: the "done" frame showed the next flight, and a previous pickup was counted) — tests test_state_at_end_of_a_decision_shows_the_finished_action, test_outcome_and_highlight_exclude_events_at_the_start RED→GREEN — cost if wrong: none
- Task 9: Ruling: the GIF output of demo.py is replaced by the two MP4s (plan); README/assets update in Task 10 — cost if wrong: README image must be regenerated
- Task 9: Step 5 (owner review) pending; continuing with Part C meanwhile
- Task 11: Ruling: added make_envs(n_envs, rng) -> (envs, state) to create the collection state; collect stores reward/10 and appends finished-day metrics to state['finished'] — needed by Task 12 logging — cost if wrong: none
- style commit 2e7afad (line length) applied to m2-network (branch moved to it) and m3-ppo
Task 11: complete (commits 4100a5d..2e70dee, tests: python -m pytest -q → 186 passed in 41.73s)
- Task 12: result: 1M pilots (seed 11, 20 select days): set .881/.783, MLP .850/.724 vs Slack-aware .810/.659; stable KL/clip; MLP within noise of the spike
Task 12: complete (commits 2e70dee..5fd5449, tests: python -m pytest -q → 187 passed in 41.34s)
- Final: fixed checkpoint label collision — test_checkpoint_labels_are_unique RED→GREEN, suite 186/186 (m2-network)
- Final: fixed nearest-first delivery tie-break (earliest box deadline) — test_nearest_breaks_ties_by_the_earliest_box_deadline RED→GREEN (re-graded Minor→Important: baseline must match its description)
- Final: fixed BC init seeding (new_network) — test_new_network_is_reproducible RED→GREEN; checkpoints retrained (MLP high .649→.627) (re-graded Minor→Important: report numbers must be reproducible)
- Final: fixed network-choice label clipping in frames (one line per choice; 7 pending rows when shown) — test_network_choices_stay_inside_the_frame RED→GREEN (re-graded Minor→Important: visible defect in the graded video)
- Final: demo_network prints logits and value estimates — test_demo_network_runs_from_any_directory (value estimate) RED→GREEN (re-graded Minor→Important: M2 rubric asks for input/output)
- Final: minor (deferred): training CSVs and bc_history.json lack env_version
- Final: fixed PPO/demonstration seed overlap (DEMO_SEEDS 0-999, RL_SEED_RANGE 1000-99999) — test_ppo_days_avoid_demonstration_seeds RED→GREEN, suite 192/192 (m3-ppo)
- Task 9: Step 5: owner reviewed the videos (2026-10-23): OK for now; a redesign discussion is planned after all work is done
Task 9: complete (commits d18ca18..b71ba82, tests: python -m pytest -q → 186 passed in 295.11s (0:04:55))
- Task 10: verified: git archive HEAD → fresh conda-forge env (python 3.11.17) + pip install -r requirements.txt (torch 2.14.1+cpu) + pip install -e .; demo_network.py, evaluate.py --methods slack --loads low, demo.py all succeed; archive has no docs/tests/internal scripts
Task 10: complete (commits d18ca18..90beccc, tests: python -m pytest -q → 186 passed in 298.98s (0:04:58))
