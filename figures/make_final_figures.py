"""
make_final_figures.py
=======================
Regenerates every main-text and supplementary figure fresh, from the
current validated CSVs, with clear final filenames -- run this ONCE
everything is confirmed clean (final_contamination_sweep.py all OK) to
produce the definitive figure set for the manuscript. Do not mix these
outputs with any previously-generated PNG from earlier in the analysis.

Usage:
    python make_final_figures.py --close-config summary_config.json \
        --far-config summary_config_far.json \
        --spectra-prefix-1qmp-300 2DIR_ilemet_1qmp_300K \
        --spectra-prefix-1qmp-340 2DIR_ilemet_1qmp_340K \
        --spectra-prefix-1dz3-300 2DIR_ilemet_1dz3_300K \
        --spectra-prefix-1dz3-340 2DIR_ilemet_1dz3_340K \
        --outdir final_figures
"""
import argparse
import json
import os
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
p.add_argument("--close-config", default="summary_config.json")
p.add_argument("--far-config", default="summary_config_far.json")
p.add_argument("--spectra-prefix-1qmp-300", default=None)
p.add_argument("--spectra-prefix-1qmp-340", default=None)
p.add_argument("--spectra-prefix-1dz3-300", default=None)
p.add_argument("--spectra-prefix-1dz3-340", default=None)
p.add_argument("--outdir", default="final_figures")
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)


def load_summary(config_path):
    with open(config_path) as f:
        entries = json.load(f)
    rows = []
    for e in entries:
        df = pd.read_csv(e["csv"])
        rows.append({
            "pair": e["pair"], "state": e["state"], "temp": e["temp"],
            "n": len(df),
            "J_AB_mean": df["J_AB_cm1"].mean(),
            "J_AB_sem": df["J_AB_cm1"].std() / np.sqrt(len(df)),
            "R_AB_mean": df["R_AB_Angstrom"].mean(),
        })
    return pd.DataFrame(rows)


def bar_figure(table, title, out_path, ylabel="J$_{AB}$ (cm$^{-1}$)"):
    pairs = table["pair"].unique()
    states = sorted(table["state"].unique())
    temps = sorted(table["temp"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.8 * len(pairs), 3.9), squeeze=False)
    axes = axes[0]
    for ax, pair in zip(axes, pairs):
        sub = table[table["pair"] == pair]
        x = np.arange(len(states))
        width = 0.35
        for i, temp in enumerate(temps):
            vals, errs = [], []
            for state in states:
                row = sub[(sub["state"] == state) & (sub["temp"] == temp)]
                vals.append(row["J_AB_mean"].values[0] if len(row) else np.nan)
                errs.append(row["J_AB_sem"].values[0] if len(row) else 0)
            ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=4, label=temp)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(pair)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Temperature", loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---- Figure 2: Close-state summary ----
close_table = load_summary(args.close_config)
close_table.to_csv(f"{args.outdir}/Table1_close_state.csv", index=False)
bar_figure(close_table, "Close-state coupling", f"{args.outdir}/Figure2_close_summary.png")

# ---- Figure 3: Far-state summary ----
far_table = load_summary(args.far_config)
far_table.to_csv(f"{args.outdir}/Table1_far_state.csv", index=False)
bar_figure(far_table, "Far-state coupling (control)", f"{args.outdir}/Figure3_far_summary.png")

# ---- Table 1: merged ----
close_table["ensemble_state"] = "Close"
far_table["ensemble_state"] = "Far"
merged = pd.concat([close_table, far_table], ignore_index=True)
merged = merged[["pair", "state", "temp", "ensemble_state", "n", "J_AB_mean", "J_AB_sem", "R_AB_mean"]]
merged.to_csv(f"{args.outdir}/Table1_full_summary.csv", index=False)
print(f"Saved: {args.outdir}/Table1_full_summary.csv")

# ---- Figure 4: combined 2D-IR spectra, one multi-panel figure ----
prefixes = {
    "1QMP, 300K": args.spectra_prefix_1qmp_300,
    "1QMP, 340K": args.spectra_prefix_1qmp_340,
    "1DZ3, 300K": args.spectra_prefix_1dz3_300,
    "1DZ3, 340K": args.spectra_prefix_1dz3_340,
}
available = {k: v for k, v in prefixes.items() if v is not None}

if available:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    axes = axes.flatten()
    for ax, (label, prefix) in zip(axes, available.items()):
        try:
            absorptive = np.load(f"{prefix}_absorptive.npy")
            w1 = np.load(f"{prefix}_w1.npy")
            w3 = np.load(f"{prefix}_w3.npy")
            W1, W3 = np.meshgrid(w1, w3)
            levels = np.linspace(-1, 1, 21)
            cf = ax.contourf(W1, W3, absorptive, levels=levels, cmap="RdBu_r")
            ax.contour(W1, W3, absorptive, levels=levels, colors="k", linewidths=0.3)
            ax.plot([w1.min(), w1.max()], [w1.min(), w1.max()], "k--", linewidth=0.8)
            ax.set_xlabel(r"$\omega_1$ (cm$^{-1}$)")
            ax.set_ylabel(r"$\omega_3$ (cm$^{-1}$)")
            ax.set_title(label)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f"Missing data:\n{prefix}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
    # Turn off any unused panels
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle("Ile-Met absorptive 2D-IR spectra")
    fig.tight_layout()
    out_path = f"{args.outdir}/Figure4_2dir_spectra_combined.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
else:
    print("No --spectra-prefix-* arguments given -- skipped Figure 4.")

print(f"\nAll outputs written to {args.outdir}/ -- use ONLY these files in the manuscript.")
