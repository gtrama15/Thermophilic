"""
make_figure1.py
=================
Figure 1: backbone C=O midpoint distance vs. time, for one or more
residue pairs, with the Close and Far ensemble regions directly shaded
on the curve -- shows the reader exactly how the structural ensembles
were defined, rather than requiring them to infer it from the text.

Usage (one pair):
    python make_figure1.py --csv distance_1dz3_3_28.csv --label "Ile-Met (1DZ3, 300K)" \
        --close-max 8 --far-min 85 --out final_figures/Figure1_distance_ilemet.png

Usage (both pairs, side by side -- the actual Figure 1 layout):
    python make_figure1.py \
        --csv distance_1dz3_3_28.csv --label "Ile-Met (1DZ3, 300K)" --close-max 8 --far-min 85 \
        --csv2 distance_1dz3_81_100.csv --label2 "Ile81-Ala100 (1DZ3, 300K)" --close-max2 10 --far-min2 65 \
        --out final_figures/Figure1_distance_combined.png
"""
import argparse
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
p.add_argument("--csv", required=True)
p.add_argument("--label", default="")
p.add_argument("--close-max", type=float, required=True)
p.add_argument("--far-min", type=float, required=True)
p.add_argument("--csv2", default=None)
p.add_argument("--label2", default="")
p.add_argument("--close-max2", type=float, default=None)
p.add_argument("--far-min2", type=float, default=None)
p.add_argument("--out", default="Figure1_distance.png")
p.add_argument("--ymax", type=float, default=None,
                help="Clip y-axis (Angstrom) to exclude PBC-artifact spikes from view; "
                     "leave unset to autoscale to the 99th percentile.")
args = p.parse_args()

panels = [(args.csv, args.label, args.close_max, args.far_min)]
if args.csv2:
    panels.append((args.csv2, args.label2, args.close_max2, args.far_min2))

fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.4), squeeze=False)
axes = axes[0]

for ax, (csv_path, label, close_max, far_min) in zip(axes, panels):
    df = pd.read_csv(csv_path)
    t = df["time_ps"].values
    r = df["R_AB_Angstrom"].values

    ymax = args.ymax if args.ymax is not None else np.percentile(r, 99) * 1.15

    # Shade Close and Far regions directly using the same thresholds
    # used to build the actual ensembles, so the figure is a faithful
    # visual record of the ensemble definition, not just decoration.
    close_mask = r <= close_max
    far_mask = r >= far_min

    ax.plot(t, r, "-", linewidth=0.8, color="gray", zorder=1)
    ax.scatter(t[close_mask], r[close_mask], s=6, color="#4C72B0", label=f"Close (\u2264{close_max} \u00c5)", zorder=2)
    ax.scatter(t[far_mask], r[far_mask], s=6, color="#DD8452", label=f"Far (\u2265{far_min} \u00c5)", zorder=2)

    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("R$_{AB}$ (\u00c5)")
    ax.set_title(label)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="Close"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#DD8452", markersize=8, label="Far"),
           Line2D([0], [0], color="gray", linewidth=1.5, label="Transition (excluded)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0.09, 1, 1])
fig.savefig(args.out, dpi=300)
print(f"Saved: {args.out}")
