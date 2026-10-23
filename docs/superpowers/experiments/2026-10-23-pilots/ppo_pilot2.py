"""Spike: minimal masked PPO (MLP) from scratch; does on-time rate rise at a given load?"""
import sys, time
import numpy as np, torch, torch.nn as nn
from medical_uav.env import MedicalUAVEnv
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS
torch.set_num_threads(4); torch.manual_seed(0)
LOAD = sys.argv[1] if len(sys.argv) > 1 else "low"
TOTAL = int(float(sys.argv[2])) if len(sys.argv) > 2 else 500_000
N_ENV, N_STEP, EPOCHS, MB = 16, 128, 4, 512

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.pi = nn.Sequential(nn.Linear(403,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,26))
        s.v = nn.Sequential(nn.Linear(403,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,1))
    def forward(s, o, m): return s.pi(o).masked_fill(~m, -1e9), s.v(o).squeeze(-1)

net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-4, eps=1e-5)
envs = [MedicalUAVEnv(load="medium" if LOAD == "mixed" else LOAD) for _ in range(N_ENV)]
rng = np.random.default_rng(0)
obs, masks = [], []
LOADS = ("low", "medium", "high")
def reset(e):
    opts = {"load": LOADS[int(rng.integers(3))]} if LOAD == "mixed" else None
    return e.reset(seed=int(rng.integers(1 << 30)), options=opts)
for e in envs:
    o, i = reset(e); obs.append(o); masks.append(i["action_mask"])
obs, masks = np.array(obs, np.float32), np.array(masks, bool)

def evaluate(n=10):
    r = []
    for s in list(DEV_SEEDS)[:n]:
        env = MedicalUAVEnv(load="medium" if LOAD == "mixed" else LOAD); o, i = env.reset(options={"demand_seed": s}); done = False
        while not done:
            with torch.no_grad():
                lg, _ = net(torch.as_tensor(o)[None], torch.as_tensor(i["action_mask"])[None])
            o, _, done, _, i = env.step(int(lg.argmax()))
        r.append(episode_metrics(env.sim)["on_time_rate"])
    return np.mean(r)

steps, t0, upd = 0, time.time(), 0
print(f"load {LOAD}: untrained greedy on-time {evaluate():.3f}", flush=True)
while steps < TOTAL:
    B = dict(o=[], m=[], a=[], lp=[], v=[], r=[], d=[])
    for _ in range(N_STEP):
        with torch.no_grad():
            lg, v = net(torch.as_tensor(obs), torch.as_tensor(masks))
            dist = torch.distributions.Categorical(logits=lg); a = dist.sample()
        B["o"].append(obs.copy()); B["m"].append(masks.copy()); B["a"].append(a.numpy()); B["lp"].append(dist.log_prob(a).numpy()); B["v"].append(v.numpy())
        rew, dn = np.zeros(N_ENV, np.float32), np.zeros(N_ENV, bool)
        for k, e in enumerate(envs):
            assert masks[k, a[k]]
            o, r, term, trunc, i = e.step(int(a[k])); rew[k], dn[k] = r / 10, term or trunc
            if dn[k]:
                o, i = reset(e)
            obs[k], masks[k] = o, i["action_mask"]
        B["r"].append(rew); B["d"].append(dn)
    steps += N_ENV * N_STEP
    with torch.no_grad():
        _, last_v = net(torch.as_tensor(obs), torch.as_tensor(masks))
    V = np.array(B["v"]); R = np.array(B["r"]); Dn = np.array(B["d"]); adv = np.zeros_like(R); gae = 0
    nv = last_v.numpy()
    for t in reversed(range(N_STEP)):
        nonterm = 1.0 - Dn[t]
        delta = R[t] + nv * nonterm - V[t]
        gae = delta + 0.95 * nonterm * gae
        adv[t] = gae; nv = V[t]
    ret = adv + V
    f = lambda x: torch.as_tensor(np.array(x).reshape(N_STEP * N_ENV, *np.array(x).shape[2:]))
    o_, m_, a_, lp_, adv_, ret_ = f(B["o"]), f(B["m"]), f(B["a"]), f(B["lp"]), f(adv), f(ret)
    frac = 1 - steps / TOTAL
    for g in opt.param_groups: g["lr"] = 3e-4 * max(frac, 0.05)
    ent_coef = 0.01 * max(frac, 0.1)
    idx = np.arange(len(a_))
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for i0 in range(0, len(idx), MB):
            b = idx[i0:i0 + MB]
            lg, v = net(o_[b], m_[b]); dist = torch.distributions.Categorical(logits=lg)
            ratio = (dist.log_prob(a_[b]) - lp_[b]).exp()
            ad = adv_[b]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            pl = -torch.min(ratio * ad, ratio.clamp(0.8, 1.2) * ad).mean()
            vl = 0.5 * ((v - ret_[b]) ** 2).mean()
            loss = pl + 0.5 * vl - ent_coef * dist.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    upd += 1
    if upd % 25 == 0:
        print(f"{steps:>8d} decisions  {time.time()-t0:5.0f}s  dev on-time {evaluate():.3f}", flush=True)
torch.save(net.state_dict(), f"ppo_{LOAD}.pt"); print("saved")
