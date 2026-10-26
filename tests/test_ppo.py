"""Numerical boundaries and continuous training invariants, beyond forward/backward."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from medical_uav.agents.networks import MLPActorCritic
from medical_uav.algorithms.ppo import PPOConfig, PPOUpdater, clipped_policy_loss
from medical_uav.algorithms.returns import generalized_advantage
from medical_uav.env import MedicalUAVEnv
from medical_uav.evaluation.evaluate import evaluate, selection_score
from medical_uav.training.collector import Collector
from medical_uav.training.io import rng_state
from medical_uav.training.trainer import TrainConfig, load_checkpoint, train


def test_gae_keeps_env_axes_and_distinguishes_terminal_truncated_and_rollout_end():
    rewards = torch.tensor([[1., 10.], [2., 20.], [3., 30.]])
    values = torch.tensor([[.5, 5.], [.6, 6.], [.7, 7.]])
    next_values = torch.tensor([[.6, 6.], [.7, 7.], [.8, 8.]])
    terminated = torch.tensor([[False, False], [True, False], [False, False]])
    truncated = torch.tensor([[False, True], [False, False], [False, False]])
    adv, targets = generalized_advantage(rewards, values, next_values, terminated, truncated, 1, 1)
    assert torch.allclose(adv, torch.tensor([[2.5, 11.], [1.4, 52.], [3.1, 31.]]))
    assert torch.allclose(targets, adv + values)
    # A huge next value at a TRUE terminal must have no effect.
    next_values[1, 0] = 1e6
    assert torch.equal(adv, generalized_advantage(rewards, values, next_values, terminated, truncated, 1, 1)[0])


def test_clipped_objective_for_positive_and_negative_advantages():
    old = torch.zeros(2)
    loss, ratio, _ = clipped_policy_loss(torch.tensor([1.5, .5]).log(), old,
                                         torch.tensor([1., -1.]), .2)
    assert loss.item() == pytest.approx(-.2)
    assert torch.allclose(ratio, torch.tensor([1.5, .5]))


class CapturingTimeLimit(gym.wrappers.TimeLimit):
    def step(self, action):
        result = super().step(action)
        self.final_observation = result[0].copy()
        return result


def test_collector_bootstraps_final_observation_before_time_limit_reset():
    torch.manual_seed(7)
    model = MLPActorCritic()
    collector = Collector(1, 5, env_factory=lambda: CapturingTimeLimit(MedicalUAVEnv(), 1))
    try:
        buffer, records = collector.collect(model, 1)
        with torch.no_grad():
            _, final_value = model(torch.from_numpy(collector.envs[0].final_observation))
        assert buffer.truncated[0, 0] and not buffer.terminated[0, 0]
        assert buffer.next_value[0, 0].item() == pytest.approx(final_value.item(), abs=1e-6)
        buffer.finish()
        assert buffer.returns[0, 0].item() == pytest.approx(
            (buffer.reward[0, 0] + final_value).item(), abs=1e-6)
        assert records[0]["truncated"] and not records[0]["complete_day"]
    finally:
        collector.close()


def test_continuous_collection_update_and_fixed_raw_value_targets():
    torch.manual_seed(13)
    model = MLPActorCritic()
    updater = PPOUpdater(model, PPOConfig(epochs=2, minibatch_size=64), 13)
    collector = Collector(2, 13)
    try:
        episodes = []
        for _ in range(3):
            buffer, rows = collector.collect(model, 128)
            episodes.extend(rows)
            buffer.finish()
            targets, old_probs, observations = buffer.returns.clone(), buffer.old_log_prob.clone(), buffer.obs.clone()
            actor, critic = model.actor.weight.detach().clone(), model.critic.weight.detach().clone()
            result = updater.update(buffer)
            assert result["initial_ratio_max_error"] < 1e-5
            assert result["optimizer_steps"] > 1
            assert not torch.equal(actor, model.actor.weight)
            assert not torch.equal(critic, model.critic.weight)
            assert torch.equal(targets, buffer.returns)
            assert torch.equal(old_probs, buffer.old_log_prob)
            assert torch.equal(observations, buffer.obs)
            assert torch.allclose(buffer.returns, buffer.advantage + buffer.old_value)
        assert len(episodes) > 2
        assert {r["env_index"] for r in episodes} == {0, 1}
        assert all(r["invalid_actions"] == 0 and r["complete_day"] for r in episodes)
        assert sum(collector.transitions_by_load.values()) == 768
        assert sum(collector.started_episodes_by_load.values()) == len(episodes) + 2
        assert any(collector.decisions)  # uncompleted days survive rollout boundaries
        for row in episodes:
            assert sum(row["action_counts"].values()) == row["decisions"]
            assert sum(x["released"] for x in row["type_counts"].values()) == row["released"]
            assert sum(row["segment_minutes"].values()) == pytest.approx(row["episode_minutes"])
    finally:
        collector.close()


def test_stale_old_probability_is_rejected():
    model = MLPActorCritic()
    collector = Collector(1)
    try:
        buffer, _ = collector.collect(model, 8)
        buffer.finish()
        buffer.old_log_prob += 0.1
        with pytest.raises(RuntimeError, match="ratio"):
            PPOUpdater(model).update(buffer)
    finally:
        collector.close()


def test_evaluation_preserves_rng_model_and_random_baseline_day_identity():
    model = MLPActorCritic()
    before = rng_state()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    rows, _ = evaluate(model, ("ppo", "random", "slack"), (1, 2), "train", ("medium",))
    after = rng_state()
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert model.training
    reversed_rows, _ = evaluate(policies=("random",), seeds=(2, 1), split="train", loads=("medium",))
    random_rows = [r for r in rows if r["policy"] == "random"]
    assert random_rows == list(reversed(reversed_rows))
    for day in (1, 2):
        assert len({r["released"] for r in rows if r["demand_seed"] == day}) == 1


def test_checkpoint_resume_restores_optimizer_and_explicitly_starts_new_days(tmp_path):
    config = TrainConfig(total_steps=128, num_envs=2, rollout_steps=32, eval_split="train",
                         evaluate_initial=False, evaluate_baselines=False)
    first = train(config, tmp_path / "first")
    saved = load_checkpoint(tmp_path / "first/last.pt")
    assert saved["steps"] == first["transitions"] == 128
    assert (tmp_path / "first/checkpoint_000000128.pt").is_file()
    assert first["training_seconds"] > 0
    assert first["started_counts_include_history"]
    optimizer_steps = [v["step"].item() for v in saved["optimizer_state_dict"]["state"].values()]
    assert optimizer_steps and min(optimizer_steps) > 0
    next_config = replace(config, total_steps=192)
    resumed = train(next_config, tmp_path / "resumed", tmp_path / "first/last.pt")
    end = load_checkpoint(tmp_path / "resumed/last.pt")
    assert resumed["transitions"] == 192 and resumed["updates"] == 3
    assert min(v["step"].item() for v in end["optimizer_state_dict"]["state"].values()) > min(optimizer_steps)
    assert resumed["resume_mode"] == "new_train_days"
    assert sum(end["collector"]["transitions_by_load"].values()) == 192
    train(next_config, tmp_path / "resumed_again", tmp_path / "first/last.pt")
    again = load_checkpoint(tmp_path / "resumed_again/last.pt")
    assert all(torch.equal(v, again["model_state_dict"][k]) for k, v in end["model_state_dict"].items())
    assert not (tmp_path / "resumed/best.pt").exists()  # Train cannot select a Dev best.
    with pytest.raises(ValueError, match="fresh"):
        train(config, tmp_path / "first")


def test_selection_is_equal_load_daily_average_with_reward_tiebreaker():
    summary = {"split": "dev", "demand_seeds": list(range(100_000, 100_040)),
               "policies": {"ppo": {"by_load": {k: {"episodes": 40} for k in ("low", "medium", "high")},
                                     "mean_on_time_rate": .5, "mean_reward": 1.}}}
    assert selection_score(summary) == (.5, 1.)
    summary["policies"]["ppo"]["by_load"].pop("high")
    with pytest.raises(ValueError):
        selection_score(summary)


@pytest.mark.parametrize("kwargs", [{"total_steps": 123}, {"eval_split": "test"},
                                    {"loads": ("unknown",)}, {"train_seeds": (100_000,)}])
def test_training_rejects_ambiguous_budgets_and_split_leakage(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs).validate()


def test_evaluation_rejects_days_from_the_wrong_split():
    with pytest.raises(ValueError):
        evaluate(policies=("slack",), split="dev", seeds=(0,))
