"""
shortrange_sensitivity.py
===========================
Tests whether the phosphorylation/temperature coupling trend is robust
across the sampled Close-state distance range, rather than being driven
by only the very closest configurations -- bins each Close-state ensemble
by R_AB into 1-Angstrom intervals and reports the mean J_AB per bin per
condition, plus the trend with the shortest bin excluded.

Uses ONLY columns already present in existing coupling CSVs (R_AB_Angstrom,
J_AB_cm1) -- no re-computation or pipeline changes needed.

Usage:
    python shortrange_sensitivity.py --config summary_config.json \
        --out final_figures/FigureS4_shortrange_sensitivity.png \
        --table-out final_figures/TableS3_shortrange_sensitivity.csv
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
p.add_argument("--bin-width", type=float, default=1.0)
p.add_argument("--out", default="FigureS4_shortrange_sensitivity.png")
p.add_argument("--table-out", default="TableS3_shortrange_sensitivity.csv")
args = p.parse_args()

with open(args.config) as f:
    entries = json.load(f)

rows = []
binned_data = {}  # (pair, state, temp) -> {bin_center: (mean, n)}

for e in entries:
    df = pd.read_csv(e["csv"])
    r_min, r_max = df["R_AB_Angstrom"].min(), df["R_AB_Angstrom"].max()
    bins = np.arange(np.floor(r_min), np.ceil(r_max) + args.bin_width, args.bin_width)
    df["bin"] = pd.cut(df["R_AB_Angstrom"], bins)

    key = (e["pair"], e["state"], e["temp"])
    binned_data[key] = {}

    for interval, group in df.groupby("bin", observed=True):
        if len(group) < 3:
            continue
        bin_center = interval.mid
        mean_j = group["J_AB_cm1"].mean()
        rows.append({
            "pair": e["pair"], "state": e["state"], "temp": e["temp"],
            "R_AB_bin_low": round(interval.left, 1), "R_AB_bin_high": round(interval.right, 1),
            "n": len(group), "J_AB_mean": round(mean_j, 4),
            "J_AB_sem": round(group["J_AB_cm1"].std() / np.sqrt(len(group)), 4),
        })
        binned_data[key][bin_center] = (mean_j, len(group))

table = pd.DataFrame(rows)
table.to_csv(args.table_out, index=False)
print(table.to_string(index=False))
print(f"\nSaved: {args.table_out}")

# --- Trend robustness check: does excluding the shortest bin change the
#     phosphorylation-state ordering (1QMP vs 1DZ3) at each temperature? ---
print("\n" + "="*70)
print("TREND ROBUSTNESS CHECK (excluding shortest-distance bin)")
print("="*70)
pairs = table["pair"].unique()
for pair in pairs:
    for temp in table["temp"].unique():
        sub = table[(table["pair"] == pair) & (table["temp"] == temp)]
        states = sub["state"].unique()
        if len(states) < 2:
            continue
        print(f"\n{pair}, {temp}:")
        for state in states:
            s = sub[sub["state"] == state].sort_values("R_AB_bin_low")
            if len(s) < 2:
                print(f"  {state}: insufficient bins for trend check")
                continue
            full_mean = np.average(s["J_AB_mean"], weights=s["n"])
            excl_shortest = s.iloc[1:]
            excl_mean = np.average(excl_shortest["J_AB_mean"], weights=excl_shortest["n"]) if len(excl_shortest) else np.nan
            print(f"  {state}: full-range weighted mean={full_mean:.3f}, "
                  f"excl.-shortest-bin mean={excl_mean:.3f}, diff={abs(full_mean-excl_mean):.3f}")

# --- Figure: J_AB vs R_AB bin, one panel per pair, lines per condition ---
fig, axes = plt.subplots(1, len(pairs), figsize=(3.9*len(pairs), 3.9), squeeze=False)
axes = axes[0]
for ax, pair in zip(axes, pairs):
    for key, bins in binned_data.items():
        if key[0] != pair:
            continue
        centers = sorted(bins.keys())
        means = [bins[c][0] for c in centers]
        label = f"{key[1]}, {key[2]}"
        ax.plot(centers, means, "o-", label=label, markersize=5)
    ax.set_xlabel("R$_{AB}$ (Å)")
    ax.set_ylabel("J$_{AB}$ (cm$^{-1}$)")
    ax.set_title(pair)
    ax.axhline(0, color="k", linewidth=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0.16, 1, 1])
fig.savefig(args.out, dpi=250)
print(f"\nSaved: {args.out}")
