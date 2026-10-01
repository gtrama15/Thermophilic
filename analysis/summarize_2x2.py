"""
summarize_2x2.py
==================
Final synthesis: pulls together Close-state coupling results across
phosphorylation state (1QMP vs 1DZ3) and temperature (300K vs 340K) for one
or more residue pairs, into a single summary table and bar chart -- the
main comparison figure for the paper.

Usage:
    python summarize_2x2.py --config summary_config.json --out summary_2x2.png

Config file format (JSON): a list of entries, each with pair label,
phospho state, temperature, and the coupling CSV to summarize:
[
  {"pair": "Ile-Met", "state": "1QMP (phospho)", "temp": "300K", "csv": "ensemble_2_27_close_rep1.csv"},
  {"pair": "Ile-Met", "state": "1QMP (phospho)", "temp": "340K", "csv": "ensemble_2_27_close_340K.csv"},
  {"pair": "Ile-Met", "state": "1DZ3 (unphospho)", "temp": "300K", "csv": "ensemble_1dz3_3_28_close.csv"},
  {"pair": "Ile-Met", "state": "1DZ3 (unphospho)", "temp": "340K", "csv": "ensemble_1dz3_340K_3_28_close.csv"},
  {"pair": "Ile81-Ala100", "state": "1QMP (phospho)", "temp": "300K", "csv": "ensemble_81_100_close.csv"},
  {"pair": "Ile81-Ala100", "state": "1QMP (phospho)", "temp": "340K", "csv": "ensemble_81_100_close_340K.csv"},
  {"pair": "Ile81-Ala100", "state": "1DZ3 (unphospho)", "temp": "300K", "csv": "ensemble_1dz3_81_100_close.csv"},
  {"pair": "Ile81-Ala100", "state": "1DZ3 (unphospho)", "temp": "340K", "csv": "ensemble_1dz3_340K_81_100_close.csv"}
]
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--out", default="summary_2x2.png")
p.add_argument("--table-out", default="summary_2x2_table.csv")
args = p.parse_args()

with open(args.config) as f:
    entries = json.load(f)

rows = []
for e in entries:
    df = pd.read_csv(e["csv"])
    mean_J = df["J_AB_cm1"].mean()
    sem_J = df["J_AB_cm1"].std() / np.sqrt(len(df))
    rows.append({
        "pair": e["pair"], "state": e["state"], "temp": e["temp"],
        "n": len(df), "J_AB_mean": mean_J, "J_AB_sem": sem_J,
        "R_AB_mean": df["R_AB_Angstrom"].mean(),
    })

table = pd.DataFrame(rows)
table.to_csv(args.table_out, index=False)
print(table.to_string(index=False))
print(f"\nSaved table: {args.table_out}")

pairs = table["pair"].unique()
fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
axes = axes[0]

states = sorted(table["state"].unique())
temps = sorted(table["temp"].unique())
colors = {temps[0]: "#4C72B0", temps[1]: "#DD8452"} if len(temps) == 2 else {}

for ax, pair in zip(axes, pairs):
    sub = table[table["pair"] == pair]
    x = np.arange(len(states))
    width = 0.35
    for i, temp in enumerate(temps):
        vals, errs = [], []
        for state in states:
            row = sub[(sub["state"] == state) & (sub["temp"] == temp)]
            if len(row):
                vals.append(row["J_AB_mean"].values[0])
                errs.append(row["J_AB_sem"].values[0])
            else:
                vals.append(np.nan)
                errs.append(0)
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=4,
               label=temp, color=colors.get(temp))
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=15, ha="right")
    ax.set_ylabel("J$_{AB}$ (cm$^{-1}$)")
    ax.set_title(pair)
    ax.legend(title="Temperature")

fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"Saved figure: {args.out}")
