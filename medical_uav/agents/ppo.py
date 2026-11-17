"""Proximal Policy Optimization with action masking, written for this project.

One update: run 16 environments for 128 decisions each with the current policy, estimate
advantages with GAE (gamma = 1, one step per decision), then improve the policy and the value
function for 4 epochs on minibatches of 512 with the clipped PPO objective.
"""
from dataclasses import dataclass

import numpy as np
import torch

from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.metrics import episode_metrics
from medical_uav.splits import RL_SEED_RANGE

LOADS = ("low", "medium", "high")  # every new day draws one of these uniformly
REWARD_SCALE = 10.0  # the learner sees reward / 10


@dataclass
class RolloutBuffer:
    """What the policy saw and did over n_steps decisions in n_envs environments, [T, N, ...]."""
    obs: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    log_prob: np.ndarray
    value: np.ndarray
    reward: np.ndarray  # already divided by REWARD_SCALE
    done: np.ndarray  # True if the day ended after this decision
    last_value: np.ndarray  # value of the state after the last decision, [N]


def _new_day(env, rng, maps=None):
    """Reset env to a random online-RL training day at a random load, on a map drawn from
    `maps` if given (else the env's map); return (obs, mask)."""
    options = {"demand_seed": int(rng.integers(*RL_SEED_RANGE)),
               "load": LOADS[int(rng.integers(len(LOADS)))]}
    if maps is not None:
        options["map_seed"] = int(maps[int(rng.integers(len(maps)))])
    obs, info = env.reset(options=options)
    return obs, info["action_mask"]


def make_envs(n_envs: int, rng, config: EnvConfig | None = None, maps=None) -> tuple[list, dict]:
    """Create n_envs environments (with the given settings, e.g. a reward variant), each started
    on a random day (on a random map from `maps` if given); return them and the collection
    state (current obs and masks, the map pool, and the results of finished days)."""
    envs = [MedicalUAVEnv(config=config) for _ in range(n_envs)]
    starts = [_new_day(env, rng, maps) for env in envs]
    state = {"obs": np.array([s[0] for s in starts]), "mask": np.array([s[1] for s in starts]),
             "maps": maps, "finished": []}
    return envs, state


def collect(envs, net, state: dict, n_steps: int, rng) -> RolloutBuffer:
    """Run the policy for n_steps decisions in every environment, sampling actions."""
    n = len(envs)
    shape = (n_steps, n)
    buf = RolloutBuffer(obs=np.zeros(shape + state["obs"].shape[1:], np.float32),
                        mask=np.zeros(shape + state["mask"].shape[1:], bool),
                        action=np.zeros(shape, np.int64), log_prob=np.zeros(shape, np.float32),
                        value=np.zeros(shape, np.float32), reward=np.zeros(shape, np.float32),
                        done=np.zeros(shape, bool), last_value=np.zeros(n, np.float32))
    for t in range(n_steps):
        buf.obs[t], buf.mask[t] = state["obs"], state["mask"]
        with torch.no_grad():
            logits, value = net(torch.as_tensor(state["obs"]), torch.as_tensor(state["mask"]))
            # masked actions get ~0 probability; training samples, the trained agent takes the top
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        buf.action[t], buf.log_prob[t] = action.numpy(), dist.log_prob(action).numpy()
        buf.value[t] = value.numpy()
        for k, env in enumerate(envs):
            a = int(action[k])
            assert state["mask"][k, a], f"sampled invalid action {a}"  # env would silently WAIT
            obs, reward, done, _, info = env.step(a)
            buf.reward[t, k], buf.done[t, k] = reward / REWARD_SCALE, done
            if done:
                state["finished"].append(episode_metrics(env.sim))
                obs, mask = _new_day(env, rng, state["maps"])  # the next day starts at once
            else:
                mask = info["action_mask"]
            state["obs"][k], state["mask"][k] = obs, mask
    with torch.no_grad():
        _, last = net(torch.as_tensor(state["obs"]), torch.as_tensor(state["mask"]))
    buf.last_value = last.numpy()
    return buf


def compute_gae(rewards, values, dones, last_value, lam: float = 0.95):
    """Generalized advantage estimation with gamma = 1, one step per decision.
    A finished day contributes no value after its last decision, and nothing crosses a reset.
    Returns (advantages, returns), both [T, N]."""
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + next_value * not_done - values[t]  # one-step TD error
        gae = delta + lam * not_done * gae  # discounted sum of TD errors (lambda only, gamma = 1)
        adv[t] = gae
        next_value = values[t]
    return adv, adv + values  # value targets = advantage + old value


def ppo_update(net, opt, buf: RolloutBuffer, adv, ret, clip: float = 0.2, epochs: int = 4,
               minibatch: int = 512, ent_coef: float = 0.01, rng=None) -> dict:
    """Improve the policy and the value function on one rollout; return mean statistics."""
    flat = lambda x: torch.as_tensor(x.reshape(-1, *x.shape[2:]))
    obs, mask, action = flat(buf.obs), flat(buf.mask), flat(buf.action)
    old_log_prob = flat(buf.log_prob)
    adv, ret = flat(adv), flat(ret)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    index = np.arange(len(action))
    for _ in range(epochs):
        rng.shuffle(index)
        for i in range(0, len(index), minibatch):
            b = torch.as_tensor(index[i:i + minibatch])
            logits, value = net(obs[b], mask[b])
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action[b])
            ratio = (log_prob - old_log_prob[b]).exp()  # new / old probability of the taken action
            a = (adv[b] - adv[b].mean()) / (adv[b].std() + 1e-8)  # normalised per minibatch
            # clipped objective: no gain from moving the ratio beyond 1 ± clip
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            value_loss = 0.5 * ((value - ret[b]) ** 2).mean()
            entropy = dist.entropy().mean()
            # value weight 0.5 (0.25 x squared error overall); the entropy bonus keeps exploring
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # limits single large steps
            opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append((old_log_prob[b] - log_prob).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > clip).float().mean().item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def day_returns(buf: RolloutBuffer):
    """The decisions of days that finished within the rollout and their exact return to go
    (gamma = 1, scaled rewards), as flat tensors (obs, mask, return); the unfinished tail of
    each environment is left out because its return is not known."""
    obs, mask, ret = [], [], []
    for k in range(buf.reward.shape[1]):
        ends = np.flatnonzero(buf.done[:, k])
        if len(ends) == 0:
            continue
        last = ends[-1] + 1
        to_go, g = np.zeros(last, np.float32), 0.0
        for t in reversed(range(last)):
            g = buf.reward[t, k] + (0.0 if buf.done[t, k] else g)
            to_go[t] = g
        obs.append(buf.obs[:last, k])
        mask.append(buf.mask[:last, k])
        ret.append(to_go)
    return tuple(torch.as_tensor(np.concatenate(x)) for x in (obs, mask, ret))


VALUE_PARTS = ("value_head", "critic")  # the value part of each network (SetNet, MLPNet)


def warm_up_value(net, envs, state: dict, rng, rollouts: int = 40, epochs: int = 4,
                  n_steps: int = 128, minibatch: int = 512, check_steps: int = 512) -> dict:
    """Train only the value part of a network on returns of its own policy, with every other
    parameter frozen, so a network trained by imitation starts PPO with a sensible critic and
    unchanged action probabilities. Progress is measured on a held-out rollout of check_steps
    decisions: the mean squared error and explained variance of the values against the exact
    returns of the days that finished in it, before and after the warm-up."""
    value_params = [p for name, p in net.named_parameters() if name.startswith(VALUE_PARTS)]
    frozen = [p for name, p in net.named_parameters() if not name.startswith(VALUE_PARTS)]
    for p in frozen:
        p.requires_grad_(False)
    check_obs, check_mask, check_ret = day_returns(collect(envs, net, state, check_steps, rng))

    def check():
        with torch.no_grad():
            error = check_ret - net(check_obs, check_mask)[1]
        return (error ** 2).mean().item(), 1 - (error.var() / check_ret.var()).item()

    error_before, explained_before = check()
    opt = torch.optim.Adam(value_params, lr=3e-4, eps=1e-5)
    for _ in range(rollouts):
        buf = collect(envs, net, state, n_steps, rng)
        _, ret = compute_gae(buf.reward, buf.value, buf.done, buf.last_value)
        obs = torch.as_tensor(buf.obs.reshape(-1, buf.obs.shape[-1]))
        mask = torch.as_tensor(buf.mask.reshape(-1, buf.mask.shape[-1]))
        target = torch.as_tensor(ret.reshape(-1))
        index = np.arange(len(target))
        for _ in range(epochs):
            rng.shuffle(index)
            for i in range(0, len(index), minibatch):
                b = torch.as_tensor(index[i:i + minibatch])
                loss = ((net(obs[b], mask[b])[1] - target[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    for p in frozen:
        p.requires_grad_(True)
    state["finished"] = []  # warm-up days are not training results
    error_after, explained_after = check()
    return {"value_error_before": error_before, "value_error_after": error_after,
            "explained_variance_before": explained_before,
            "explained_variance_after": explained_after,
            "decisions": (rollouts * n_steps + check_steps) * len(envs)}
