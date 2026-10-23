"""Fixed seed ranges for training, screening, model selection and testing."""

TRAIN_SEED_RANGE = (0, 100_000)  # demand seeds drawn by env.reset by default; end excluded
SCREEN_SEEDS = range(100_000, 100_040)  # calibration, environment checks and debugging
SELECT_SEEDS = range(100_040, 100_140)  # checkpoint and model selection
TEST_SEEDS = range(200_000, 200_300)  # final reported results only
TRAINING_SEEDS = (11, 22, 33, 44, 55)  # seeds of independent training runs
