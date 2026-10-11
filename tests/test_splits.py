from medical_uav import splits


def test_splits_disjoint():
    lo, hi = splits.TRAIN_SEED_RANGE
    for seed in list(splits.DEV_SEEDS) + list(splits.TEST_SEEDS):
        assert not lo <= seed < hi
    assert not set(splits.DEV_SEEDS) & set(splits.TEST_SEEDS)


def test_default_demand_seeds_come_from_the_training_range():
    from medical_uav.env import MedicalUAVEnv

    env = MedicalUAVEnv()
    lo, hi = splits.TRAIN_SEED_RANGE
    for seed in range(10):
        env.reset(seed=seed)
        assert lo <= env.demand_seed < hi
    env.reset(seed=3, options={"demand_seed": None})  # None means "draw one"
    assert lo <= env.demand_seed < hi
