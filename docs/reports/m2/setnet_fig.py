"""Figure 1 of the M2 report: the set network (SetNet in medical_uav/agents/networks.py), drawn
left to right from the observation to the outputs.

    python docs/reports/m2/setnet_fig.py   # writes setnet.png next to this file
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, EDGE = "#1f2a3d", "#6b778a", "#4a5568"
ENCODER, ATTENTION, HEAD, VALUE, OUTPUT = "#dbe7f6", "#fbe3c8", "#ddefd9", "#eadcf2", "#fff1cf"

fig, ax = plt.subplots(figsize=(10, 2.9), dpi=260)
ax.set_xlim(0, 10)
ax.set_ylim(-0.25, 2.75)
ax.axis("off")


def box(x, y, w, h, color, title="", sub=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=0.7))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 0.09, title, ha="center", va="center", fontsize=8.4,
                weight="bold", color=INK)
        ax.text(x + w / 2, cy - 0.12, sub, ha="center", va="center", fontsize=7, color=INK)
    elif title:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=8.4, weight="bold",
                color=INK)


def note(x, y, text, ha="center", va="bottom"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=6.8, color=MUTED, style="italic")


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EDGE,
                                 shrinkA=0, shrinkB=0))


for x, text in ((1.07, "ENCODERS · input [B, 452]"), (3.65, "ATTENTION"), (6.42, "HEADS"),
                (9.0, "OUTPUT")):
    ax.text(x, 2.62, text, ha="center", va="center", fontsize=7, weight="bold", color=MUTED)

top, mid, low, bh = 2.0, 1.2, 0.4, 0.42  # rows: orders, drone state, cargo
r = bh / 2
box(0.0, top, 2.15, bh, ENCODER, "order encoder (shared)", "12 slots × 27 → 64 → 64")
box(0.0, mid, 2.15, bh, ENCODER, "UAV encoder", "global 80 → 64")
box(0.0, low, 2.15, bh, ENCODER, "cargo encoder (shared)", "3 slots × 16 → 64")

box(2.65, 0.3, 2.0, 1.42, ATTENTION)
ax.text(3.65, 1.47, "4-head attention", ha="center", va="center", fontsize=8.4, weight="bold",
        color=INK)
ax.text(3.65, 1.04, "Q: UAV encoding\nK, V: orders, cargo and\na learned null token;\n"
        "empty slots masked", ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.25)
ax.text(3.65, 0.5, "context: 64 ⊕ 64 = 128", ha="center", va="center", fontsize=7,
        weight="bold", color=INK)

box(5.2, top, 2.45, bh, HEAD, "PICK scorer (shared)", "order + context: 192 → 64 → 1")
box(5.2, mid, 2.45, bh, HEAD, "other actions", "context: 128 → 64 → 14")
box(5.2, low, 2.45, bh, VALUE, "value head", "context + mean order: 192 → 64 → 1")

box(8.05, mid, 1.95, 1.22, OUTPUT)
ax.text(9.02, 1.98, "logits [B, 26]", ha="center", va="center", fontsize=8.4, weight="bold",
        color=INK)
ax.text(9.02, 1.56, "12 PICK, 6 DELIVER,\n6 CHARGE, RETURN, WAIT;\nmask: forbidden → −10⁹",
        ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.3)
box(8.05, low, 1.95, bh, OUTPUT, "value [B]")

# encoders to attention and to the PICK scorer
arrow((2.15, top + r), (5.2, top + r))
note(3.65, top + r + 0.04, "each order's encoding")
arrow((2.15, top + 0.12), (2.85, 1.72))
note(2.62, 1.86, "K, V", ha="left")
arrow((2.15, mid + r), (2.65, mid + r))
note(2.5, mid + r + 0.03, "Q")
arrow((2.15, low + r), (2.65, low + r))
note(2.5, low + r + 0.03, "K, V")
# the value head also takes the mean of the occupied order encodings (zero if none)
ax.plot([2.15, 2.32, 2.32, 5.65], [top + 0.03, top + 0.03, 0.08, 0.08], color=EDGE, lw=0.8,
        ls=(0, (3, 2)))
arrow((5.65, 0.08), (5.65, low))
note(3.95, 0.04, "mean of occupied order encodings", va="top")
# attention to the heads, heads to the outputs
arrow((4.65, 1.72), (5.4, top))
note(4.88, 1.72, "context", ha="left", va="top")
arrow((4.65, mid + r), (5.2, mid + r))
arrow((4.65, low + r), (5.2, low + r))
arrow((7.65, top + r), (8.05, 1.98))
arrow((7.65, mid + r), (8.05, mid + r))
arrow((7.65, low + r), (8.05, low + r))

fig.savefig(Path(__file__).with_name("setnet.png"), bbox_inches="tight", pad_inches=0.02,
            facecolor="white")
