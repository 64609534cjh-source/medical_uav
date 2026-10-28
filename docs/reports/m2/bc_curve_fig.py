"""Behaviour-cloning learning curves for the M2 report: validation agreement with the teacher per
epoch, both networks, both teachers (checkpoints/bc_history.json and bc_rollout_history.json).

    python docs/reports/m2/bc_curve_fig.py <checkpoints folder>   # writes bc_curve.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NETS = {"set": ("Set network", "#eb6834"), "mlp": ("MLP", "#2a78d6")}
# teacher file, group label and where it sits inside the plot (above its two curves)
TEACHERS = {"bc_history.json": ("imitating Slack-aware", 98.5), "bc_rollout_history.json": ("imitating rollout", 80.5)}
LABEL_DY = {"set": 1.8, "mlp": -1.8}  # keeps the two end labels of a teacher apart

folder = Path(sys.argv[1])
fig, ax = plt.subplots(figsize=(4.3, 1.1), dpi=300)
for file, (group, group_y) in TEACHERS.items():
    history = json.load(open(folder / file))
    for kind, (name, color) in NETS.items():
        acc = [100 * e["val_acc"] for e in history[kind]]
        loss = [e["val_loss"] for e in history[kind]]
        best = min(range(len(loss)), key=loss.__getitem__)  # the epoch kept: lowest validation loss
        ax.plot(range(1, len(acc) + 1), acc, color=color, lw=1.2)
        ax.plot(best + 1, acc[best], "o", ms=4, mfc="white", mec=color, mew=1.2, zorder=3)
        ax.text(len(acc) + 2.5, acc[-1] + LABEL_DY[kind], f"{name} {acc[best]:.1f}%",
                color=INK, fontsize=7, va="center")
    ax.text(12, group_y, group, color=MUTED, fontsize=7, style="italic", va="bottom")

ax.set_xlim(1, 100)
ax.set_ylim(60, 103)
ax.set_xticks([1, 25, 50, 75, 100])
ax.set_yticks([60, 70, 80, 90, 100])
ax.set_xlabel("epoch", fontsize=7, color=MUTED, labelpad=1)
ax.set_ylabel("agreement (%)", fontsize=7, color=MUTED)
ax.tick_params(labelsize=7, colors=MUTED, length=0)
ax.grid(axis="y", color=GRID, lw=0.6)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.savefig(Path(__file__).with_name("bc_curve.png"), bbox_inches="tight", pad_inches=0.03,
            facecolor="white")
