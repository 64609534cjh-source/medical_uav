# Roadmap to the Final (M2 → M3 → Final) — Design

Group 17 · 2026-10-23 · authoritative for everything after M1. It supersedes the milestone sections (§8, §10–§12) of `2026-10-10-medical-uav-design.md`; the M1 environment there stays valid except for the changes in §3. The M2 specifics are in `2026-10-23-m2-network-design.md`.

**Ownership.** The owner handed the technical design to Claude and Codex ("treat it as your own project, think long-term, do meaningful work"). Decisions below were reached by Claude and Codex (gpt-6-astra, high) after independent proposals and one convergence round, using the pilot evidence in §9. The owner is told the results; items marked **owner** wait for the owner.

## 1. Research question and story

**Can scratch PPO learn better batching and charging decisions than hand-made rules and online planning, and how much do imitation, reward choice and network structure contribute?**

The final report should tell three connected findings:
1. **Objective design matters:** a reward that values late rescues gives less timely service (§9.1, §9.4).
2. **Planning shows achievable headroom** above the strongest rule (Slack-aware).
3. **Learning turns experience into a dispatcher:** PPO from scratch beats the rules; imitation and architecture are measured as controlled contributions, not assumed.

Pre-registered targets (dev → confirmed on locked test days):
- Useful improvement: ≥ +3 points over Slack-aware at medium load, paired 95 % CI above zero; no more than 2 points worse at high, 1 point at low.
- Planning comparison: within 2 points of the visible-order planner, or above it, with the inference-speed difference measured.
- Explanation: identify which decisions account for differences (batching, charging timing, repositioning, giving up hopeless work).
A result that only matches a teacher is reported as distillation; a failure is reported as a failure.

## 2. Non-goals

Clinical triage claims; real airspace or dynamics; multiple UAVs; hyper-parameter searches beyond the stated controls; tuning anything on test days.

## 3. Environment v2 (frozen before any training data)

| item | decision |
|---|---|
| **Reward** | **compromise:** +1 per on-time delivery, +0.1 per late delivery, −0.1 per expired order, −0.001 per battery % used. With every order resolved this equals `1.1·N_on + 0.2·N_late − 0.1·N_released − 0.001·E`, i.e. it prioritises timely service while still rewarding late rescues. Coefficients frozen; on-time-only and the M1 reward are kept as ablations. |
| **Observation** | add the full 7×7 node-to-node travel-time matrix (÷ `TRAVEL_SCALE`) to the global block: global 31 → **80**, observation 403 → **452**. Constant on map 0; needed for the map-transfer study and matches the proposal's "route distances and travel times". |
| **Unchanged** | map 0 primary, 12 pending + 3 cargo slots, `Discrete(26)` incl. RETURN, non-interruptible charging to 60/80/100 %, expiry at 2× limit, expired cargo handling, masks, WAIT always valid, loads 0.04/0.09/0.12. Baseline calibration (Table 1 of M1) is unaffected because the rules do not use the reward or observation. |
| **Versioning** | M1 stays at tag `m1-submission-v1`. The environment records a schema/reward version string used in every checkpoint and result file. |

## 4. Seeds and evaluation protocol

- Training: demand seeds in [0, 100 000); demonstrations and online RL use separate blocks; the actual seeds are recorded.
- **Screen** 100 000–100 039 (environment and debugging checks, M1 calibration days); **select** 100 040–100 139 (checkpoint and model selection); **test** 200 000–200 299, locked until methods and selection rules are frozen.
- Training seeds 11, 22, 33 for controlled comparisons; 44, 55 added for the primary agent.
- One checkpoint per training seed, the same for all loads: maximise the medium/high mean on-time rate on the selection days, subject to low-load ≥ 97 % and no order type more than 3 points below Slack-aware. If none qualifies, report it.
- Metrics: mean daily on-time rate and pooled counts; served, expired, per-type on-time with denominators; return; energy, charging share, batching (boxes per stop), WAIT/RETURN share, decision latency.
- Statistics: paired bootstrap over whole days (10 000 resamples) against Slack-aware and the planner; training-seed spread reported separately; per load, never only averaged.
- The shared evaluator ships with every submission from M2 on.

## 5. Agents

- **Networks:** set/attention network (main candidate) and tanh MLP (matched control), both outputting masked logits for 26 actions and a value. Set network as in the M2 design, with the global encoder `80 → 64`. MLP `452 → 256 → 256` actor and separate critic. Switch the primary agent to the MLP only if three matched seeds show it ahead by > 2 points on the medium/high mean or the set network stays unstable.
- **PPO (own code):** start from the pilot settings exactly — 16 envs × 128 decisions, 4 epochs, minibatch 512, clip 0.2, Adam 3e-4 with linear decay (floor 5 %), entropy 0.01 decaying (floor 10 %), GAE λ 0.95, γ = 1, reward ÷ 10, advantage normalisation per minibatch, gradient clip 0.5, value loss **0.25 × MSE** (the pilot's effective factor; do not silently change). Collection-time masks and log-probs are stored; invalid actions are asserted never to be sent; GAE never crosses a reset. One policy for all loads, load drawn uniformly per episode; no curriculum. Budget 6 M decisions per run; no automatic extension.
- **Imitation (controls):** BC from Slack-aware (also the M2 learning evidence) and BC from the visible-order planner, each 900 balanced episodes, ≤ 30 epochs, split 80/20 by demand seed; BC → PPO from both. BC-only checkpoints kept; BC → PPO evaluated at 0, 0.5, 1, 3, 6 M decisions; demonstration and pre-training cost reported separately. Before BC → PPO, warm up the value head on on-policy data with the policy frozen. No DAgger by default.

## 6. Baselines

Random, EDF (M1 continuity); FCFS, nearest-first (proposal, M2); Slack-aware (strong rule); **visible-order rollout planner** (one-step lookahead over Slack-aware using only the 12 visible orders and cargo; passes an information audit: changing future arrivals or hidden orders never changes its choice, on 100 saved states); **scenario rollout** (final; samples future arrivals from the demand model with a separate RNG, common scenarios across candidate actions, 4 vs 16 scenarios over a 60-minute horizon, chosen on dev days within a measured p95 decision time of 1 s). Described as literature-grounded adaptations, not as "state of the art" by label.

## 7. Milestones

| dates | work | submitted |
|---|---|---|
| 10/23–25 | environment v2 (§3), seed manifest, both networks, shared evaluator, FCFS/nearest | — |
| 10/26–28 | Slack-aware BC (curves, closed loop), M2 demo, readable visualization (§8; owner reviews before submission), production PPO end to end on both networks (short runs), checkpoint save/reload, throughput; start the planner audit | — |
| **10/29–30 M2** | extracted-archive rehearsal, report | networks, BC code and checkpoints, forward/backward demo, readable visualization, evaluator, environment v2, README, 2-page report; PPO stays on its branch |
| **10/31–11/6 M3** | 3 seeds of set-network scratch PPO; MLP control; Slack-BC → PPO; planner (validated) and planner-BC → PPO; learning curves, probes, failure analysis | per the M3 rubric (to obtain): training/evaluation code, trained agents, comparisons, report |
| 11/7–11/11 | reward controls (current, on-time only); primary architecture to 5 seeds; scenario rollout; optional unseen-map study (maps as clusters) only after the core comparisons | — |
| 11/12–14 | locked test evaluation, uncertainty, per-type, latency, batching/charging case studies | — |
| 11/15–17 | final video, individual reports, clean-machine rehearsal; each member explains the formulation and one failure | — |
| **11/18–20 Final** | fixes and buffer | training + testing code, trained models, environment, README, video, individual ≤ 10-page reports |

Budget (environment decisions): debugging ≤ 3 M; primary scratch PPO 5 × 6 M; alternative architecture 3 × 6 M; Slack-BC → PPO 3 × 6 M; planner-BC → PPO 3 × 6 M; reward controls 2 × 3 × 6 M; optional mixed-map 3 × 6 M. About 123 M core decisions; measure production throughput first (pilot: 6 M in 9–27 min on 4 threads).

## 8. Visualization (in M2; owner reviews before submission)

The M1 GIF showed 12 simulated minutes per second; nobody can follow it. Planned fix, built from the existing event logs: (1) a 45–90 s explanation replay of one short decision sequence, holding each decision 2–4 s with the order, deadline, chosen action, battery/cargo consequence and outcome; (2) an accelerated day overview with a labelled playback speed; MP4 output. Final video 3–4 min: task, Slack-aware vs learned policy on the same arrivals synchronised by simulated time, one batching and one charging decision explained, aggregate results and one limitation. Owner (10/23): it must be fixed before the M2 submission; Claude and Codex build a first version, the owner reviews it and asks for changes before submitting.

## 9. Evidence so far (pilots, 2026-10-23)

Scripts and logs: `docs/superpowers/experiments/2026-10-23-pilots/` (spikes, not production code; commit a6abf17 + uncommitted spike scripts). Map 0, screen days 100 000–100 039 unless stated, one training seed.

1. **Planner reward screen** (visible orders only), on-time / served / plasma — medium: Slack .810/.906/.74; current .831/.966/.80; on-time-only .902/.923/.73; compromise .897/.946/.79. High: Slack .659/.807/.64; current .660/.879/.61; on-time-only .772/.817/.56; compromise .768/.854/.64.
2. **BC pilot** (MLP, 300 Slack-aware episodes, 15 epochs, 30 s): validation accuracy .88; closed loop (20 days) low .990 vs .995, medium .763 vs .822, high .626 vs .682. DAgger (3 rounds): medium .795 → .810 → .819 → .804, high .649 → .670 → .648 → .644 — not reliable enough to depend on.
3. **Scratch PPO pilot** (MLP, M1 reward): low 1.000 after 0.5 M; medium-only 3 M: .873 vs .810 (+6.3 [+4.7, +8.1], 33/40 days); high-only 3 M: .780 vs .659 (+12.1 [+10.3, +13.8], 39/40); one mixed-load policy 6 M: low .999, medium .886 (+7.6 [+5.8, +9.5], 36/40), high .766 (+10.7 [+9.1, +12.4], 39/40); all order types up except plasma at high (.62 vs .64); served up, expiry down.
4. **PPO reward comparison** (mixed, 6 M, same seed): current .999/.886/.766; on-time-only .998/.913/.800 (high served .846, plasma .61); compromise .997/.908/.801 (high served .864, expired .136, plasma .63).

5. **M2 BC, production code (environment v2)** — `python scripts/train_bc.py` (900 Slack-aware days, seeds 0–299 × 3 loads, 88 744 decisions, split by seed 80/20, 30 epochs): validation accuracy MLP .935, set .939 (networks initialised from seed 0). `python scripts/evaluate.py --methods random,edf,slack,fcfs,nearest --checkpoints checkpoints/bc_mlp.pt checkpoints/bc_set.pt --seeds screen`: on-time (low / medium / high) — Random .555/.263/.226; EDF .996/.406/.275; FCFS .970/.493/.321; nearest .992/.716/.532; Slack-aware .993/.810/.659; BC-MLP .992/.793/.627; BC-set .995/.795/.644 (vs Slack-aware: medium −1.6 [−2.9, −0.4] / −1.5 [−3.4, +0.2]; high −3.2 [−5.3, −0.9] / −1.5 [−3.2, +0.2]). An earlier unseeded MLP run gave .649 at high: BC's closed-loop result moves by ~2 points with the initialisation.

6. **Production PPO, 1 M decisions, seed 11** (branch `m3-ppo`, `python scripts/train_ppo.py --net {mlp,set} --seed 11 --decisions 1e6 --eval-every 2.5e5`; 4 CPU threads; greedy on the first 20 selection days): set network medium .879 → .889 → .891 → .881, high .763 → .771 → .777 → .783 (at 0.25/0.5/0.75/1 M; 267 s); MLP medium .823 → .839 → .852 → .850, high .661 → .708 → .721 → .724 (209 s). Slack-aware on the same days: .810 / .659. After 1 M decisions the set network is ahead of the MLP by 3.1 (medium) and 5.9 (high) points — one seed, to be confirmed with 3 seeds in M3. Approximate KL ≈ 0.001, clip fraction ≈ 1 %, entropy falling 0.61 → 0.22: stable updates.

7. **Scratch PPO, 3 seeds × 6 M decisions per network** (`python scripts/train_ppo.py --net {set,mlp} --seed {11,22,33} --decisions 6e6 --eval-every 5e5`, 6 runs in parallel, 55–65 min each; final checkpoints, not yet selected by the §4 rule; `python scripts/evaluate.py ... --seeds screen`, 40 screen days, file `runs/m3_screen_eval.csv`). On-time (low / medium / high): set .999/.917/.830 (seeds .923/.916/.912 medium, .842/.824/.825 high); MLP .998/.907/.808; Slack-aware .993/.810/.659; BC-set .995/.795/.644. Every set and MLP seed beats Slack-aware with paired CIs far above zero (medium +9.5 to +11.3, high +14.3 to +18.3 points). Set − MLP, seeds averaged per day: medium +1.0 [+0.4, +1.5], high +2.3 [+1.5, +3.1] → the set network stays primary (§5 rule). Per type, the set network beats Slack-aware on all four (high: plasma .70 vs .64, medicine .91 vs .74, sample .86 vs .66, supplies .81 vs .59) with fewer expiries (high .135 vs .193). For reference, the visible-order rollout spike with the compromise score reached .897 / .768 (§9.1): PPO is above it by about 2 and 6 points (planner not yet audited).

## 10. Decision log (since M1)

| date | decision | was | why |
|---|---|---|---|
| 10/23 | reward = compromise | M1: linear late reward to −0.5, expiry −0.5 | §9.1, §9.4: +2.5–3.5 on-time points over the M1 reward with PPO; same on-time as on-time-only, more served, plasma not sacrificed |
| 10/23 | observation 452 (7×7 travel times) | 403 | map-transfer study; proposal's "route distances and travel times" |
| 10/23 | primary agent = scratch PPO, set network provisional, MLP control | v2 spec: MLP first, BC → PPO as a fallback; Codex round 1: planner-BC → PPO primary | §9.3: scratch PPO already beats Slack-aware clearly |
| 10/23 | one mixed-load policy, no curriculum | v2 spec: low → medium → mixed curriculum | §9.3: the mixed policy learns as well as single-load ones |
| 10/23 | seeds: screen / select / test 300 days; training seeds 11–55 | dev 40 days, test 100 days | separate model selection from screening; more test power |
| 10/23 | baselines: + visible-order planner (audited), + scenario rollout in the final | rolling-horizon optimizer (proposal); one-step rollout (M1 report) | the M1 report promised the planner; the final asks for a strong conventional comparison |
| 10/23 | visualization fixed in M2, owner reviews before submission | GIF at 12 simulated minutes per second | TA: "too fast to understand anything"; owner: must be fixed before M2 |

## 11. Report notes (fill in as we go)

- **M2 reflections (network):** changes from the proposal — separate encoders for orders, cargo and global state; fixed heads for the 14 non-PICK actions instead of a universal action scorer; top-12 visibility; MLP kept as a control; BC used to show learning before RL. Environment changes since M1 that M2 must state: reward changed to the compromise and why (§9), observation extended to 452. Simplifications: one fixed map for the primary study; value head not trained by BC.
- **M3 reflections (agent):** scratch PPO vs imitation, reward controls, architecture choice and the selection rule; anything that failed.
- **Final:** the full chain proposal → M1 → M2 → M3 → final, using §10.
