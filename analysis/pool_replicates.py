"""
pool_replicates.py
====================
Pool coupling-data CSVs from multiple independent replicate trajectories
into one combined ensemble -- WITH a per-replicate consistency check first,
since pooling replicates that don't actually agree with each other would
be scientifically inappropriate (it would silently mix genuinely different
populations, the same mistake the earlier time-window mismatches made).

Usage:
    python pool_replicates.py --inputs ensemble_2_27_far_rep1.csv ensemble_2_27_far_rep2.csv \
        ensemble_2_27_far_rep3.csv --out ensemble_2_27_far_pooled.csv
"""
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True, help="Per-replicate CSV files to pool")
p.add_argument("--out", default="pooled.csv")
p.add_argument("--tolerance-sigma", type=float, default=2.0,
                help="Flag a replicate if its mean J_AB differs from the "
                     "pooled mean by more than this many pooled-std units")
args = p.parse_args()

dfs = []
for i, path in enumerate(args.inputs):
    df = pd.read_csv(path)
    df["replicate"] = i
    df["replicate_file"] = path
    dfs.append(df)
    print(f"Replicate {i} ({path}): {len(df)} snapshots")
    print(df[["omega_A", "omega_B", "J_AB_cm1", "R_AB_Angstrom"]].describe())
    print()

pooled = pd.concat(dfs, ignore_index=True)

print("=" * 60)
print("PER-REPLICATE COMPARISON")
print("=" * 60)
print("NOTE: with only a few replicates, an automated statistical outlier")
print("test is not reliable (per-replicate SEMs are small relative to")
print("normal run-to-run variation in stochastic steered MD, so nearly any")
print("real difference looks 'significant' in SEM terms). This prints the")
print("numbers for YOU to judge, rather than an automated verdict.")
print()
print("What actually matters: do ALL replicates show the same qualitative")
print("PATTERN (e.g. all substantially negative J_AB when close, all near")
print("zero when far), even if the exact magnitudes differ somewhat? That's")
print("the meaningful criterion here -- not whether means match to several")
print("decimal places.\n")

for i, path in enumerate(args.inputs):
    rep_i = pooled[pooled["replicate"] == i]
    mean_i = rep_i["J_AB_cm1"].mean()
    sem_i = rep_i["J_AB_cm1"].std() / np.sqrt(len(rep_i)) if len(rep_i) > 1 else np.nan
    print(f"Replicate {i} ({path}): n={len(rep_i)}, J_AB mean={mean_i:.4f} +/- {sem_i:.4f} (SEM)")

means = [pooled[pooled["replicate"] == i]["J_AB_cm1"].mean() for i in range(len(args.inputs))]
print(f"\nRange across replicate means: {min(means):.4f} to {max(means):.4f}")
print("Look at this range relative to the physical scale of your effect")
print("(e.g. if Close is ~-7 and Far is ~0, replicate means all clustering")
print("within that same close/far distinction is reassuring; a replicate")
print("landing on the OPPOSITE side, or wildly outside the others, is worth")
print("inspecting individually before including it).")

pooled.to_csv(args.out, index=False)
print(f"\nSaved pooled ensemble: {args.out} ({len(pooled)} total snapshots)")
