### Strengths

The PPO core matches the pilot: collection-time masks and log-probabilities are retained, GAE stops at episode boundaries, rewards are divided by 10 once, and the effective value-loss coefficient is **0.25 × MSE**. Reward accounting matched the v2 formula on complete low-, medium-, and high-load days.

The set network correctly splits the 452 features, masks empty attention slots, retains an always-valid null token, and preserves pending-slot permutation behaviour. Evaluation pairs results by demand seed, and training demand seeds remain outside screen/select/test ranges.

I ran **69 targeted tests successfully**, completed `demo_network.py --quick` from `/tmp`, and inspected existing video frames. I did not rerun the complete 187-test suite.

| Required focus | Finding |
|---|---|
| Only WAIT valid | Both architectures return WAIT; probabilities sum to 1. |
| Batch of one | Tests confirm logits `[1,26]`, value `[1]`. |
| GPU checkpoint → CPU | Saving CPU tensors and loading with `map_location="cpu"` are correct; existing checkpoints load. GPU numerical parity remains unverified because CUDA is unavailable. |
| Invalid evaluation action | Evaluator raises before stepping; test passes. |
| Demo from another directory | Successfully completed from `/tmp`. |
| Replay near day end | Requesting eight decisions with one remaining produces that decision through its actual end. |

The implementer’s rulings are reasonable. Moving evaluation into package code improves reuse; the reward signature change matches v2; per-day Random seeding preserves pairing; and the replay boundary fixes correctly distinguish completed actions. The test relocation, demo selection, MP4 reuse, and collection-state additions introduce no identified correctness problems.

### Issues

#### Critical (Must Fix)

None found.

#### Important (Should Fix)

1. **Checkpoint comparisons silently discard runs with identical filenames.**  
   [scripts/evaluate.py:41](/home/caijihang/projects/medical_uav/scripts/evaluate.py:41)

   Policies are keyed by `Path(path).stem`. Both production pilots produce `ckpt_01001472.pt`; passing their two paths retains only the set network. I reproduced this with the existing MLP and set checkpoints. This silently defeats architecture and training-seed comparisons.

   **Fix:** use unique labels incorporating the run directory, or accept explicit labels and reject duplicates. Test two different runs with the same checkpoint filename.

2. **PPO reuses demonstration demand seeds despite the required separate blocks.**  
   [medical_uav/agents/ppo.py:34](/home/caijihang/projects/medical_uav/medical_uav/agents/ppo.py:34)

   `_new_day` relies on the environment’s entire `[0,100000)` range, while BC uses seeds `0–299`. Repeated actual `_new_day` calls with RNG seed 11 reach demand seed **259** on reset 229. This does **not** leak screen/select/test data, but it violates roadmap §4 and compromises the intended separation between demonstration data and online experience.

   **Fix:** define disjoint demonstration and PPO demand ranges, pass PPO’s demand seed explicitly, record the actual episode seeds, and test the partition before longer runs.

#### Minor (Nice to Have)

1. **Nearest-first uses the oldest box’s deadline when breaking delivery ties.**  
   [medical_uav/baselines/heuristics.py:140](/home/caijihang/projects/medical_uav/medical_uav/baselines/heuristics.py:140)

   The shared candidate helper selects cargo by release time for FCFS, and nearest-first reuses that box’s deadline. In a reproduced line-map case, pickup and delivery both finish at minute 15; delivery contains deadlines 35 and 75, but the policy chooses a pickup due at 60 because it compares against 75.

   **Fix:** retain oldest release for FCFS, but use the earliest relevant cargo deadline for nearest-first. Add this tied-duration case. I have not established an effect on the reported map-0 results.

2. **BC seeds training after random network initialization.**  
   [scripts/train_bc.py:41](/home/caijihang/projects/medical_uav/scripts/train_bc.py:41)

   `train_bc` calls `torch.manual_seed`, but `net_class()` has already initialized the weights. The first MLP therefore cannot be reproduced from the script’s nominal training seed, weakening reproducibility of the committed learning history.

   **Fix:** seed before constructing each network and save the initialization/training seed in checkpoint metadata.

3. **Training result files omit the required environment version.**  
   [scripts/train_ppo.py:64](/home/caijihang/projects/medical_uav/scripts/train_ppo.py:64), [scripts/train_bc.py:53](/home/caijihang/projects/medical_uav/scripts/train_bc.py:53)

   PPO’s two CSVs and `bc_history.json` lack `env_version`, although checkpoints and evaluator rows contain it. Detached learning curves cannot reliably identify their reward/schema version.

   **Fix:** include `ENV_VERSION` in each result file; also retain run seed and architecture alongside PPO metrics.

4. **The top-three probability display clips ordinary action labels.**  
   [medical_uav/env/render.py:293](/home/caijihang/projects/medical_uav/medical_uav/env/render.py:293)

   All three choices are placed on one line in a 496-pixel panel. In the existing set-network explanation video at 12 seconds, the third choice extends beyond the frame and its probability is missing.

   **Fix:** allocate separate lines or wrap the choices within a measured text area.

5. **The forward demo prints output shapes, but no numerical logits or values.**  
   [scripts/demo_network.py:77](/home/caijihang/projects/medical_uav/scripts/demo_network.py:77)

   Task 8 calls for showing logits and values. The demo prints their shapes and action probabilities, but never an actual value estimate, leaving part of the policy/value demonstration invisible.

   **Fix:** print selected valid logits and the corresponding scalar value, identifying that the BC value head is untrained.

### Declined to judge

- Task 10’s README/assets update, archive rehearsal, report, tag, and submission: not listed as completed; these remain submission prerequisites.
- Final checkpoint-selection constraints and evaluation on all selection/test days: Task 12 explicitly specifies a preliminary 20-day medium/high evaluation.
- Multi-seed superiority and final performance targets: the completed work provides pilots, not the final study.
- Planner audits, planner imitation, BC→PPO value warm-up, and reward ablations: later roadmap work.
- GPU PPO collection/inference and resumable optimizer checkpoints: outside the specified CPU scratch-training workflow.
- Truncation handling for external Gym wrappers: the supplied environment always returns `truncated=False`.
- Nondefault network widths, arbitrary invalid hyperparameters, and all-false masks: outside the fixed architecture and always-valid-WAIT contract.
- Subjective presentation preferences and owner approval of the videos: reserved for the owner; the clipping finding above is an observable defect.

### Assessment

**Ready to merge? With fixes.** Correct checkpoint identity and the PPO demand-seed partition before long runs or comparative evaluation. The core learning implementation appears sound; the smaller baseline, reproducibility, and presentation issues should be addressed before the M2 submission.