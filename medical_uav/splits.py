"""Seed manifest: which demand seeds and maps are used for what."""

TRAIN_SEED_RANGE = (0, 100_000)  # default draws in env.reset (upper bound exclusive)
DEV_SEEDS = range(100_000, 100_040)  # calibration, tuning, checkpoint selection
TEST_SEEDS = range(200_000, 200_100)  # final numbers only
GENERALIZATION_MAPS = range(300_000, 300_020)
GENERALIZATION_SEEDS = range(400_000, 400_010)
