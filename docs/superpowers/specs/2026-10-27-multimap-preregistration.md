# Multi-map training: pre-registered design (2026-10-27)

Owner's request: keep the single-map agent and its results; train the same agent on many maps and compare. Written before any multi-map run started; Codex reviews it before results.

**Question.** Does training on randomly drawn maps improve on-time service on maps never seen in training, and what does it cost on map 0?

**Arms.** (A) The existing single-map PPO: set network, v2 reward, map 0, training seeds 11/22/33/44/55, selected checkpoints frozen (`outputs/m3/selected.json`). (B) Multi-map PPO: identical network, reward, PPO settings, budget (6 M decisions) and training seeds; each new training day draws its map uniformly from `TRAIN_MAP_SEEDS` (1000–1499, 500 maps), its load uniformly and its demand seed from `RL_SEED_RANGE`, as A does. Map 0, the development maps (1–10), the selection maps (100–109) and the test maps (200–209) are never trained on. Command: `python scripts/train_ppo.py --net set --seed <s> --maps train --decisions 6e6 --threads 3 --out runs/mm_set_s<s>`; the 20-day curves rotate over the selection maps.

**Selection of B.** The §4 rule unchanged, on the selection maps: every checkpoint and Slack-aware on the first 20 selection days (100040–100059) of each of the 10 selection maps at all loads (`select_checkpoints.py --maps select --days 20`); low ≥ 97 %, every type within 3 points of Slack-aware at medium and high (pooled over maps and days), highest medium/high mean. A keeps its selected checkpoints.

**Development comparison** (before the locked test), seeds averaged per map-day, B − A:
1. *Primary — unseen maps:* the development maps 1–10, first 20 selection days, medium and high (the stress-test protocol, `map_stress.py`); score = mean of medium and high on-time; 95 % bootstrap over maps. B is better if the gain is ≥ 1 point with the interval above 0; practically equal if the interval lies within ±1; else inconclusive.
2. *Secondary — map 0:* the 40 screen days; non-inferiority margin −1 point (lower bound of B − A above −1).
3. Reported alongside: each seed, per load, Slack-aware and the visible-order rollout on the same map-days.

**Locked test.** Both arms enter the final manifest: map 0 on the test days 200000–200299 and the test maps 200–209 on the first 30 test days each, medium and high; the same paired statistics. No reselection or tuning after any test result.

**Claims.** If B helps on unseen maps without a meaningful loss on map 0, report it as the more general agent; the map-0 results of A stand either way. Ten maps give limited precision; conclusions are about this map generator.
