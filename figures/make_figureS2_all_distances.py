"""
make_figureS2_all_distances.py
================================
Figure S2: full grid of distance-vs-time curves for every residue pair x
construct x temperature combination, with Close/Far ensemble regions
shaded (same convention as Figure 1). Supplementary companion to the
single representative-condition Figure 1.

Give it a JSON list of {"label": ..., "csv": ..., "close_max": ...,
"far_min": ...} entries, one per condition.

Usage:
    python make_figureS2_all_distances.py --config all_distances_config.json \
        --out final_figures/FigureS2_all_distances.png
"""
import argparse
import json
import numpy as np
import pandas as pd
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
p.add_argument("--config", required=True)
p.add_argument("--out", default="FigureS2_all_distances.png")
p.add_argument("--ncols", type=int, default=2)
args = p.parse_args()

with open(args.config) as f:
    entries = json.load(f)

ncols = args.ncols
nrows = int(np.ceil(len(entries) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.4*nrows), squeeze=False)
axes_flat = axes.flatten()

for ax, e in zip(axes_flat, entries):
    df = pd.read_csv(e["csv"])
    t = df["time_ps"].values
    r = df["R_AB_Angstrom"].values
    close_max, far_min = e["close_max"], e["far_min"]
    ymax = np.percentile(r, 99) * 1.15

    close_mask = r <= close_max
    far_mask = r >= far_min

    ax.plot(t, r, "-", linewidth=0.6, color="gray", zorder=1)
    ax.scatter(t[close_mask], r[close_mask], s=4, color="#4C72B0", zorder=2)
    ax.scatter(t[far_mask], r[far_mask], s=4, color="#DD8452", zorder=2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("R$_{AB}$ (Å)")
    ax.set_title(e["label"], fontsize=11)

for ax in axes_flat[len(entries):]:
    ax.axis("off")

fig.tight_layout()
fig.savefig(args.out, dpi=250)
print(f"Saved: {args.out}")
