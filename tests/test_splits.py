from medical_uav import splits


def test_seed_manifest():
    lo, hi = splits.TRAIN_SEED_RANGE
    sets = [set(splits.SCREEN_SEEDS), set(splits.SELECT_SEEDS), set(splits.TEST_SEEDS)]
    assert [len(s) for s in sets] == [40, 100, 300]
    for i, a in enumerate(sets):
        assert not any(lo <= seed < hi for seed in a)
        for b in sets[i + 1:]:
            assert not a & b
    assert splits.TRAINING_SEEDS == (11, 22, 33, 44, 55)


def test_default_demand_seeds_come_from_the_training_range():
    from medical_uav.env import MedicalUAVEnv

    env = MedicalUAVEnv()
    lo, hi = splits.TRAIN_SEED_RANGE
    for seed in range(10):
        env.reset(seed=seed)
        assert lo <= env.demand_seed < hi
    env.reset(seed=3, options={"demand_seed": None})  # None means "draw one"
    assert lo <= env.demand_seed < hi
