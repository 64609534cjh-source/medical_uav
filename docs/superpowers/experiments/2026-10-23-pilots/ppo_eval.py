"""Evaluate a saved pilot PPO net on all 40 dev days vs Slack-aware, with per-type and expiry stats."""
import sys
import numpy as np, torch
from medical_uav.env import MedicalUAVEnv
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS
import torch.nn as nn
torch.set_num_threads(4)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.pi = nn.Sequential(nn.Linear(403,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,26))
        s.v = nn.Sequential(nn.Linear(403,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,1))
    def forward(s, o, m): return s.pi(o).masked_fill(~m, -1e9), s.v(o).squeeze(-1)
net = Net(); net.load_state_dict(torch.load(sys.argv[1])); net.eval()
for load in sys.argv[2].split(","):
    rows = {"ppo": [], "slack": []}
    for s in DEV_SEEDS:
        for name in rows:
            env = MedicalUAVEnv(load=load); o, i = env.reset(options={"demand_seed": s}); done = False
            while not done:
                if name == "ppo":
                    with torch.no_grad():
                        a = int(net(torch.as_tensor(o)[None], torch.as_tensor(i["action_mask"])[None])[0].argmax())
                else:
                    a = slack_aware_policy(env)
                o, _, done, _, i = env.step(a)
            rows[name].append(episode_metrics(env.sim))
    d = np.array([p["on_time_rate"] - q["on_time_rate"] for p, q in zip(rows["ppo"], rows["slack"])])
    boot = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    for name, r in rows.items():
        types = " ".join(f"{k}={np.nanmean([x['on_time_by_type'][k] for x in r]):.2f}" for k in r[0]["on_time_by_type"])
        print(f"{load:6s} {name:5s} on-time {np.mean([x['on_time_rate'] for x in r]):.3f} served {np.mean([x['served_rate'] for x in r]):.3f} "
              f"expired {np.mean([x['expiry_rate'] for x in r]):.3f} charge {np.mean([x['charge_share'] for x in r]):.2f}  {types}")
    print(f"{load:6s} paired diff ppo-slack {d.mean():+.3f}  95% CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]  wins {(d>0).sum()}/{len(d)}")
