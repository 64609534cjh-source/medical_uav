"""Spike: BC (300 teacher episodes) then 3 DAgger rounds (net drives, teacher labels) with an MLP."""
import numpy as np, torch, torch.nn as nn
from medical_uav.env import MedicalUAVEnv
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS
torch.set_num_threads(4)

class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(403,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,26))
    def forward(s, o, m): return s.f(o).masked_fill(~m, -1e9)

def act(net, obs, mask):
    with torch.no_grad():
        return int(net(torch.as_tensor(obs)[None], torch.as_tensor(mask)[None]).argmax())

def collect(seeds, net=None):
    O, Mk, A = [], [], []
    for load in ("low", "medium", "high"):
        env = MedicalUAVEnv(load=load)
        for s in seeds:
            obs, info = env.reset(options={"demand_seed": s}); done = False
            while not done:
                teacher = slack_aware_policy(env)
                O.append(obs); Mk.append(info["action_mask"]); A.append(teacher)
                a = teacher if net is None else act(net, obs, info["action_mask"])
                obs, _, done, _, info = env.step(a)
    return [np.array(O, np.float32), np.array(Mk, bool), np.array(A)]

def train(data, epochs=15, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    o, m, a = map(torch.as_tensor, data); net = MLP(); opt = torch.optim.Adam(net.parameters(), 3e-4)
    idx = np.arange(len(a))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), 256):
            b = idx[i:i+256]; loss = nn.functional.cross_entropy(net(o[b], m[b]), a[b]); opt.zero_grad(); loss.backward(); opt.step()
    return net

def evaluate(net, n=20):
    out = {}
    for load in ("medium", "high"):
        r = []
        for s in list(DEV_SEEDS)[:n]:
            env = MedicalUAVEnv(load=load); obs, info = env.reset(options={"demand_seed": s}); done = False
            while not done:
                obs, _, done, _, info = env.step(act(net, obs, info["action_mask"]))
            r.append(episode_metrics(env.sim)["on_time_rate"])
        out[load] = np.mean(r)
    return out

data = collect(range(100))
net = train(data); print("BC      ", len(data[2]), "samples", evaluate(net))
for k in range(3):
    new = collect(range(1000 + 100 * k, 1100 + 100 * k), net)
    data = [np.concatenate([x, y]) for x, y in zip(data, new)]
    net = train(data); print(f"DAgger {k+1}", len(data[2]), "samples", evaluate(net))
print("Slack-aware reference (same 20 days): medium 0.822, high 0.682")
