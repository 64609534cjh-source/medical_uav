"""Fixed seed ranges for training, development, testing and generalization."""

TRAIN_SEED_RANGE = (0, 100_000)  # drawn by env.reset by default; end excluded
DEV_SEEDS = range(100_000, 100_040)  # calibration, tuning and checkpoint selection
TEST_SEEDS = range(200_000, 200_100)  # final reported results only
GENERALIZATION_MAPS = range(300_000, 300_020)  # unseen maps
GENERALIZATION_SEEDS = range(400_000, 400_010)  # demand seeds used on the unseen maps
