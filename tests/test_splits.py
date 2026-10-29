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
    # demonstrations and online RL use separate blocks of the training range
    lo, hi = splits.RL_SEED_RANGE
    assert all(splits.TRAIN_SEED_RANGE[0] <= s < lo for s in splits.DEMO_SEEDS)
    assert splits.TRAIN_SEED_RANGE == (0, hi)


def test_default_demand_seeds_come_from_the_training_range():
    from medical_uav.env import MedicalUAVEnv

    env = MedicalUAVEnv()
    lo, hi = splits.TRAIN_SEED_RANGE
    for seed in range(10):
        env.reset(seed=seed)
        assert lo <= env.demand_seed < hi
    env.reset(seed=3, options={"demand_seed": None})  # None means "draw one"
    assert lo <= env.demand_seed < hi


def test_final_test_days_and_maps_are_new():
    used_days = [splits.TRAIN_SEED_RANGE, (100_000, 100_140), (200_000, 200_300),
                 (300_000, 300_100), (400_000, 400_100), (500_000, 500_300), (600_000, 600_300),
                 (700_000, 710_000)]
    end = splits.FINAL_MAP_DAY0 + splits.FINAL_MAP_DAYS * len(splits.FINAL_TEST_MAP_SEEDS)
    final_days = list(splits.FINAL_TEST_SEEDS) + list(range(splits.FINAL_MAP_DAY0, end))
    assert len(set(final_days)) == 600
    assert not any(lo <= d < hi for d in final_days for lo, hi in used_days)
    used_maps = ({0} | set(splits.DEV_MAP_SEEDS) | set(splits.SELECT_MAP_SEEDS)
                 | set(splits.TEST_MAP_SEEDS) | set(splits.TRAIN_MAP_SEEDS))
    assert len(splits.FINAL_TEST_MAP_SEEDS) == 30 and not used_maps & set(splits.FINAL_TEST_MAP_SEEDS)
