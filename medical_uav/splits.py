"""Fixed seed ranges for training, development and testing."""

TRAIN_SEED_RANGE = (0, 100_000)  # drawn by env.reset by default; end excluded
DEV_SEEDS = range(100_000, 100_040)  # calibration, tuning and checkpoint selection
DEV_EXTRA_SEEDS = range(100_040, 100_140)  # development diagnostics/configuration comparison; never Test
TEST_SEEDS = range(200_000, 200_100)  # final reported results only

TEST_V2_SEEDS = range(300_000, 300_100)  # Gen2 preregistered holdout; see gen2_holdout_freeze.json
