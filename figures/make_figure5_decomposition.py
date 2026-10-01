"""
make_figure5_decomposition.py
================================
Bar chart visualizing the distance-orientation decomposition results
(Table 3b): % Distance vs % Orientation contribution for each of the
comparisons, making the "two different mechanisms" finding visually
immediate rather than requiring the reader to parse a numbers table.

Usage:
    python make_figure5_decomposition.py --out final_figures/Figure5_decomposition.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure style: figures are built near their final printed width (A4 text
# width is about 6.3 in) so fonts stay legible after insertion in the manuscript.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 13,
})

p = argparse.ArgumentParser()
p.add_argument("--out", default="Figure5_decomposition.png")
args = p.parse_args()

# Real, verified data from Table 3b
comparisons = [
    "Ile-Met\nTemperature\n(1DZ3)",
    "Ile-Met\nPhosphorylation\n(300K)",
    "Ile81-Ala100\nPhosphorylation\n(300K)",
    "Ile81-Ala100\nPhosphorylation\n(340K)",
    "Ile81-Ala100\nTemperature\n(1QMP)",
]
pct_distance = [109, 111, 12, 6, 49]
pct_orientation = [-9, -12, 84, 92, 49]
pair_group = ["Ile-Met", "Ile-Met", "Ile81-Ala100", "Ile81-Ala100", "Ile81-Ala100"]

x = np.arange(len(comparisons))
width = 0.35

fig, ax = plt.subplots(figsize=(7.6, 4.8))
bars1 = ax.bar(x - width/2, pct_distance, width, label="% Distance-driven", color="#4C72B0")
bars2 = ax.bar(x + width/2, pct_orientation, width, label="% Orientation-driven", color="#DD8452")

ax.axhline(0, color="k", linewidth=0.8)
ax.set_ylabel("Contribution to coupling difference (%)")
ax.set_title("Distance\u2013orientation decomposition of coupling differences")
ax.set_xticks(x)
ax.set_xticklabels(comparisons, fontsize=10)
ax.legend(loc="upper center", ncol=2, frameon=False)
ax.set_ylim(-26, 140)

# Label each bar with its value for readability
for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 3 if h >= 0 else -3
        ax.annotate(f"{h:.0f}%", xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va=va, fontsize=10)

# Visually separate the two residue-pair groups
ax.axvline(1.5, color="gray", linestyle=":", linewidth=1)

fig.tight_layout()
fig.savefig(args.out, dpi=300)
print(f"Saved: {args.out}")
