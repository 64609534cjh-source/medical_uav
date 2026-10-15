"""Fixed seed ranges for training, development and testing."""

TRAIN_SEED_RANGE = (0, 100_000)  # drawn by env.reset by default; end excluded
DEV_SEEDS = range(100_000, 100_040)  # calibration, tuning and checkpoint selection
TEST_SEEDS = range(200_000, 200_100)  # final reported results only
