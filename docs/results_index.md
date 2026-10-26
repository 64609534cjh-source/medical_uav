# Discussion evidence index

Start with `artifacts/checkpoint_manifest.json` and `python scripts/discussion.py verify`.
Two committed archives contain original selected checkpoints and existing evidence, without machine-specific downloads. Extraction restores their original relative `outputs/...` paths; different local files are never overwritten.

| Evidence | Models / source | Protocol and report |
|---|---|---|
| M3 MLP | five Dev-selected best, per-model training SHA in manifest | `configs/final_test_protocol.json`, `results/m3_summary.md`, `results/final_test.md` |
| M4 structured | five Dev-selected best | same original Test protocol; `results/m4_summary.md` |
| M5 late LR | five Dev-selected best | original Test and Test-v2 protocols; `results/m5_summary.md` |
| Gen2 destination (default) | five Dev-selected best; training `705806aba80a06a992286464e31a363f698d726d` | `configs/gen2_test_v2_protocol.json`, `results/gen2_summary.md` |
| Gen2 attention | five Dev-selected best; retained unclear/negative result | same Test-v2 protocol/report |
| Gen3 urgency / coordination | five best each, source `8d5f1d3945f6e4086be4e108e227ee7d70d32776` | `configs/gen3_test_v3_protocol.json`, `results/gen3_summary.md` |
| Guarded inference | original weights, source baseline `d050c4c100b86bdb6676d57f400a24c9bc80e66b`; file hash in manifest | `results/gen3_inference.json`, `results/gen3_final_audit.json`; 4,800 previous full Dev days |
| New R1 development audit | existing M5/destination Dev-extra records; no Test rerun | `results/discussion_failure_audit.md`, `results/discussion_traces.json.gz` |

The model archive has **35 distinct selected checkpoint files**, including all five defaults. The evidence archive includes 288 files: all three historical Test cohorts with rule baselines, selected-run config/source manifests/summaries/update and evaluation logs, and Gen3 Dev-extra failure diagnostics. Source code is available in the Git history at the recorded commits. Checkpoint config fingerprints use sorted, compact JSON; source manifests retain their original bytes.

Run `python scripts/discussion.py table` to reproduce three separate historical mean/SD tables from full-day rows. Run `python scripts/audit_discussion_failures.py` and `python scripts/trace_discussion.py` after verification to reproduce R1 artifacts. Existing full statistical reports retain crossed learner/day paired intervals and all types; the lightweight table command does not claim to recompute every bootstrap interval or all historical plots.

Every intermediate/last checkpoint and full training episode stream remains in the developer's original local run directories; these large intermediate files are **not** included in this selected-evidence archive. Archived update/evaluation logs reproduce the reported learning/selection evidence without requiring full training reruns. Older scripts needing other intermediate paths may require the original runs. No local-only file is required by the discussion quick start.

The default CPU timing is the existing isolated single-thread i7-12800H benchmark: batch1 canonical 270.435/290.245µs p50/p95; guarded 212.484/226.709µs. These are medians of per-model round-median quantiles, not pooled p95s. Batch8 and all three repeated rounds are retained in `gen3_inference.json`. Do not combine timings from different historical benchmark protocols.

AI disclosure: Codex assisted with packaging, provenance checks, Dev-only diagnosis, executable reproduction helpers and draft documentation. Human review and independent reproduction remain required before the planned prerelease.
