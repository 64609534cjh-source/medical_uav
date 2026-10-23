"""Spike: behaviour cloning of Slack-aware with an MLP; held-out accuracy and closed-loop on-time."""
import time
import numpy as np, torch, torch.nn as nn
from medical_uav.env import MedicalUAVEnv
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS
torch.set_num_threads(4)

def collect(n_per_load):
    O, Mk, A, E = [], [], [], []
    ep = 0
    for load in ("low", "medium", "high"):
        env = MedicalUAVEnv(load=load)
        for s in range(n_per_load):
            obs, info = env.reset(options={"demand_seed": s}); done = False
            while not done:
                a = slack_aware_policy(env)
                O.append(obs); Mk.append(info["action_mask"]); A.append(a); E.append(ep)
                obs, _, done, _, info = env.step(a)
            ep += 1
    return np.array(O, np.float32), np.array(Mk, bool), np.array(A), np.array(E)

class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(403,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,26))
    def forward(s, o, m): return s.f(o).masked_fill(~m, -1e9)

t0=time.time(); O, Mk, A, E = collect(100); print("samples", len(A), "episodes", E.max()+1, f"{time.time()-t0:.0f}s")
rng = np.random.default_rng(0); eps = rng.permutation(E.max()+1); val_eps = set(eps[:len(eps)//5])
v = np.array([e in val_eps for e in E])
o, m, a = map(torch.as_tensor, (O, Mk, A))
net = MLP(); opt = torch.optim.Adam(net.parameters(), 3e-4); lossf = nn.CrossEntropyLoss()
tr = np.flatnonzero(~v); va = np.flatnonzero(v)
for epoch in range(15):
    rng.shuffle(tr)
    for i in range(0, len(tr), 256):
        b = tr[i:i+256]; loss = lossf(net(o[b], m[b]), a[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (net(o[va], m[va]).argmax(1) == a[va]).float().mean().item()
    print(f"epoch {epoch+1} train_loss {loss.item():.3f} val_acc {acc:.3f}")
def run(load, seed, use_net):
    env = MedicalUAVEnv(load=load); obs, info = env.reset(options={"demand_seed": seed}); done = False
    while not done:
        if use_net:
            with torch.no_grad():
                act = int(net(torch.as_tensor(obs)[None], torch.as_tensor(info["action_mask"])[None]).argmax())
        else:
            act = slack_aware_policy(env)
        obs, _, done, _, info = env.step(act)
    return episode_metrics(env.sim)["on_time_rate"]
for load in ("low", "medium", "high"):
    seeds = list(DEV_SEEDS)[:20]
    bc = [run(load, s, True) for s in seeds]; sl = [run(load, s, False) for s in seeds]
    print(f"{load:6s} closed-loop on-time  BC-MLP {np.mean(bc):.3f}  Slack-aware {np.mean(sl):.3f}  (20 dev days)")
