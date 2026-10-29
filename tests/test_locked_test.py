import sys

sys.path.insert(0, "scripts")
import locked_test  # noqa: E402


def test_final_test_maps_each_have_their_own_ten_days():
    maps = locked_test.blocks("test")["maps"]
    days = [d for _, seeds in maps for d in seeds]
    assert len(maps) == 30 and len(days) == len(set(days)) == 300
    assert maps[0] == (2000, range(800_300, 800_310))
    assert len(locked_test.blocks("test")["map0"][0][1]) == 300


def test_campaign_skips_the_horizon_at_low_load(tmp_path):
    manifest = {"entrants": {"slack": {"policy": "slack"}, "horizon": {"policy": "horizon"}},
                "skip": locked_test.SKIP,
                "days": {"select": locked_test.describe(locked_test.blocks("select"))}}
    todo = locked_test.expected(manifest, "select", 2, tmp_path)
    count = lambda label, load: sum(1 for j in todo if j[1] == label and j[4] == load)
    assert count("slack", "low") == (1 + 10) * 2  # map 0 and ten selection maps, two days each
    assert count("horizon", "low") == 0 and count("horizon", "high") == 22
    assert len({j[6] for j in todo}) == len(todo)
