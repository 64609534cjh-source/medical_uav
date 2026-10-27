"""Registered readings of the rolling-horizon optimizer on the 40 screen days (medium, high).

docs/superpowers/specs/2026-10-27-rolling-horizon-preregistration.md. Run from the main tree's
root after runs/horizon_screen.csv exists.
"""
import csv
from collections import defaultdict

import numpy as np

from medical_uav.env.config import ORDER_TYPES
from medical_uav.evaluation import paired_ci

PPO = ["m3_set_s11/ckpt_06000640", "m3_set_s22/ckpt_05500928", "m3_set_s33/ckpt_04501504",
       "m3_set_s44/ckpt_06000640", "m3_set_s55/ckpt_05001216"]
rows = defaultdict(dict)  # (method, load) -> {day: row}
for f in ("runs/m3_screen_eval_all.csv", "runs/scenario_screen.csv", "runs/horizon_screen.csv"):
    for r in csv.DictReader(open(f)):
        rows[r["method"].replace("runs/", ""), r["load"]][int(r["seed"])] = r
fmt = lambda c: f"{c[0] * 100:+6.2f} [{c[1] * 100:+6.2f}, {c[2] * 100:+6.2f}]"
for load in ("medium", "high"):
    days = sorted(rows["foresight", load])
    on = {m: np.array([float(rows[m, load][d]["on_time_rate"]) for d in days])
          for m in ["slack", "rollout", "scenario16", "horizon", "foresight"] + PPO}
    on["PPO"] = np.mean([on[c] for c in PPO], axis=0)
    print(f"\n{load} load, {len(days)} days: mean daily on-time")
    for m in ("slack", "rollout", "scenario16", "horizon", "foresight", "PPO"):
        line = f"  {m:11s} {on[m].mean():.3f}"
        if m in ("horizon", "foresight"):
            ms = [float(rows[m, load][d]["decision_ms_mean"]) for d in days]
            p95 = [float(rows[m, load][d]["decision_ms_p95"]) for d in days]
            pooled = [sum(int(rows[m, load][d][f"on_time_count_{t.name}"]) for d in days)
                      / sum(int(rows[m, load][d][f"released_{t.name}"]) for d in days)
                      for t in ORDER_TYPES]
            line += (f"  served {np.mean([float(rows[m, load][d]['served_rate']) for d in days]):.3f}"
                     f"  {np.mean(ms):.0f} ms/decision (max day p95 {max(p95):.0f})  "
                     + " ".join(f"{t.name} {v:.3f}" for t, v in zip(ORDER_TYPES, pooled)))
        print(line)
    print(f"  1. horizon - rollout      {fmt(paired_ci(on['horizon'] - on['rollout']))}")
    print(f"     horizon - scenario16   {fmt(paired_ci(on['horizon'] - on['scenario16']))}")
    print(f"  2. PPO - horizon          {fmt(paired_ci(on['PPO'] - on['horizon']))}  (margin -2)")
    print(f"  3. foresight - PPO        {fmt(paired_ci(on['foresight'] - on['PPO']))}")
    print(f"     foresight - horizon    {fmt(paired_ci(on['foresight'] - on['horizon']))}")
