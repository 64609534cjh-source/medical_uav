# Codex: review of the final-phase code (2026-10-27, astra high)

All points handled the same day: information contract (authoritative mask checked in the planning copy, observable-view test, candidate independence, ties by tolerance set, expired-cargo and end-of-window tests, full-precision reading recorded as a deviation); diagnostics (per-type service, battery at each peak, slot overflow share); roadmap wording (intervention effect, hypotheses); pickup metrics skip unloading of expired boxes; verdict requires the registered energy changes and enough top-ups (fewer than 0.5 per day counts as failed, set before any new result).

**Critical: none.**

**Important**

- **The information audit falls short of registration.** [scenario.py:62](/home/caijihang/projects/medical_uav/medical_uav/baselines/scenario.py:62) copies exact simulator values and original IDs, recomputes the root mask, and accepts the environment rather than observation/mask inputs. I found no actual future/hidden-order read, but this is an additional protocol deviation. [test_scenario.py:91](/home/caijihang/projects/medical_uav/tests/test_scenario.py:91) changes hidden counts/destinations without verifying equal observations. **Fix:** implement the registered interface, preserve the authoritative mask, and add equal-observation, poisoned-attribute, candidate-permutation and boundary tests.

- **Registered diagnostics are missing.** [analyze_behaviour.py:74](/home/caijihang/projects/medical_uav/scripts/analyze_behaviour.py:74) omits per-type service and separate battery values for each peak; [evaluate.py:59](/home/caijihang/projects/medical_uav/scripts/evaluate.py:59) supplies no overflow-frequency/conditional-performance diagnostics. **Fix:** retain and report these before claiming the development analysis is complete.

- **Interpretations exceed the evidence.** [roadmap:111](/home/caijihang/projects/medical_uav/docs/superpowers/specs/2026-10-23-roadmap-to-final.md:111)’s “bound the dependence” is unsupported: masking estimates an intervention effect, not a bound or mediated share. [roadmap:113](/home/caijihang/projects/medical_uav/docs/superpowers/specs/2026-10-23-roadmap-to-final.md:113)’s plasma tradeoff and four-scenario noise explanations are plausible hypotheses, not demonstrated causes. **Fix:** state “consistent with”; remove the bound claim.

**Minor**

- **Disposal breaks pickup diagnostics.** [metrics.py:135](/home/caijihang/projects/medical_uav/medical_uav/metrics.py:135) counts `fly → dispose → load` as an in-place pickup; [metrics.py:85](/home/caijihang/projects/medical_uav/medical_uav/metrics.py:85) misses `charge → dispose → load` as charge-then-pickup. Both reproduced. **Fix:** account for intervening disposal segments, preferably using action boundaries, and add regression cases.

- **Verdict logic incompletely implements manipulation checks.** [energy_contrasts.py:47](/home/caijihang/projects/medical_uav/scripts/energy_contrasts.py:47) declares “no practically meaningful benefit” even when peak charging increases; override frequency never enters the verdict. **Fix:** require the registered intended energy changes and handle few overrides explicitly. Current CSV verdicts remain correct.

- **Tolerance ties can choose the wrong index.** [scenario.py:129](/home/caijihang/projects/medical_uav/medical_uav/baselines/scenario.py:129): scores `{0:0.9e-9, 1:1.8e-9, Slack:0}` select 1, although 0 is within tolerance of the maximum. **Fix:** compute the maximum first, form its tolerance set, then prefer Slack or the lowest index.

The remaining checks found **none**: energy boundaries/reset/masking, arrival distributions, common scenarios, terminal completion—including WAITs before sampled arrivals—daily evaluation ordering/state, serial timing boundaries, battery interpolation, peak counts/charging overlap, empty kilometres and futile pickups.

Both energy CSVs reproduce §9.13’s numerical conclusions. Both scenario versions fail the plasma gate; retaining 16 **conditionally on latency feasibility** is registered handling. Available latency logs are incomplete, so certification remains pending. Twenty read-only-compatible tests passed; three file-writing tests were excluded. No files changed.
