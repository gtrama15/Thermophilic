"""
make_figureS3_distributions.py
================================
Figure S3: small-multiples grid of J_AB distributions across all
conditions, for a given ensemble state (Close or Far) -- supports the
convergence/contamination-check claims in Methods by showing the actual
distribution shape for every condition in one place.

Reuses the same config format as summarize_2x2.py / summary_config.json.

Usage:
    python make_figureS3_distributions.py --config summary_config.json \
        --title "Close-state J_AB distributions" \
        --out final_figures/FigureS3_close_distributions.png
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
p.add_argument("--title", default="")
p.add_argument("--out", default="FigureS3_distributions.png")
p.add_argument("--ncols", type=int, default=2)
args = p.parse_args()

with open(args.config) as f:
    entries = json.load(f)

ncols = args.ncols
nrows = int(np.ceil(len(entries) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.5*nrows), squeeze=False)
axes_flat = axes.flatten()

for ax, e in zip(axes_flat, entries):
    df = pd.read_csv(e["csv"])
    j = df["J_AB_cm1"].values
    ax.hist(j, bins=30, color="#4C72B0", alpha=0.85)
    ax.axvline(j.mean(), color="k", linestyle="--", linewidth=1)
    label = f"{e['pair']}\n{e['state']}, {e['temp']}"
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("J$_{AB}$ (cm$^{-1}$)", fontsize=11)
    ax.tick_params(labelsize=10)
    x0, x1 = ax.get_xlim()
    left = (j.mean() - x0) / (x1 - x0) > 0.5   # put the note on the emptier side
    ax.text(0.03 if left else 0.97, 0.95, f"n={len(df)}\nmean={j.mean():.3f}",
            transform=ax.transAxes, fontsize=9.5, va="top", ha="left" if left else "right")

for ax in axes_flat[len(entries):]:
    ax.axis("off")

if args.title:
    fig.suptitle(args.title)
fig.tight_layout()
fig.savefig(args.out, dpi=250)
print(f"Saved: {args.out}")
