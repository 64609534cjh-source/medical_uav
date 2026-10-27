"""Fixed seed ranges for training, screening, model selection and testing."""

TRAIN_SEED_RANGE = (0, 100_000)  # demand seeds drawn by env.reset by default; end excluded
DEMO_SEEDS = range(0, 1_000)  # training days reserved for recorded demonstrations (BC)
RL_SEED_RANGE = (1_000, 100_000)  # training days for online RL; end excluded
SCREEN_SEEDS = range(100_000, 100_040)  # calibration, environment checks and debugging
SELECT_SEEDS = range(100_040, 100_140)  # checkpoint and model selection
TEST_SEEDS = range(200_000, 200_300)  # final reported results only
TRAINING_SEEDS = (11, 22, 33, 44, 55)  # seeds of independent training runs

# maps (map seeds); map 0 is the map of every result unless stated
TRAIN_MAP_SEEDS = range(1_000, 1_500)  # drawn per day by multi-map training
DEV_MAP_SEEDS = range(1, 11)  # unseen-map development checks (the stress test)
SELECT_MAP_SEEDS = range(100, 110)  # checkpoint selection of multi-map runs
TEST_MAP_SEEDS = range(200, 210)  # final reported results only
