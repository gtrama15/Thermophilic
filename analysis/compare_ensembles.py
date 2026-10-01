"""
compare_ensembles.py
=====================
Direct, statistically honest comparison of two coupling-data ensembles
(e.g. "H-bond present" vs "H-bond broken"), BEFORE trusting a 2D-IR
spectral comparison built from them.

Usage:
    python compare_ensembles.py --a ensemble_hbond_present.csv --b ensemble_hbond_broken.csv \
        --label-a "Present" --label-b "Broken"
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--label-a", default="A")
p.add_argument("--label-b", default="B")
p.add_argument("--out", default="ensemble_comparison.png")
args = p.parse_args()

dfA = pd.read_csv(args.a)
dfB = pd.read_csv(args.b)

print(f"{args.label_a}: {len(dfA)} snapshots")
print(dfA[["omega_A", "omega_B", "J_AB_cm1", "R_AB_Angstrom"]].describe())
print(f"\n{args.label_b}: {len(dfB)} snapshots")
print(dfB[["omega_A", "omega_B", "J_AB_cm1", "R_AB_Angstrom"]].describe())

# Convergence check: compare first half vs second half of EACH ensemble --
# if the mean/std shift substantially between halves, that ensemble likely
# hasn't converged yet (needs more snapshots) and any A-vs-B comparison
# built on it is not yet trustworthy.
print("\n--- Convergence check (first half vs second half, each ensemble) ---")
for label, df in [(args.label_a, dfA), (args.label_b, dfB)]:
    half = len(df) // 2
    first, second = df.iloc[:half], df.iloc[half:]
    print(f"{label}: omega_A mean first-half={first['omega_A'].mean():.2f}  "
          f"second-half={second['omega_A'].mean():.2f}  "
          f"(diff={abs(first['omega_A'].mean()-second['omega_A'].mean()):.2f} cm^-1)")
    print(f"{label}: J_AB mean first-half={first['J_AB_cm1'].mean():.3f}  "
          f"second-half={second['J_AB_cm1'].mean():.3f}  "
          f"(diff={abs(first['J_AB_cm1'].mean()-second['J_AB_cm1'].mean()):.3f} cm^-1)")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

bins = np.linspace(min(dfA["omega_A"].min(), dfB["omega_A"].min()),
                    max(dfA["omega_A"].max(), dfB["omega_A"].max()), 30)
axes[0,0].hist(dfA["omega_A"], bins=bins, alpha=0.6, label=args.label_a, density=True)
axes[0,0].hist(dfB["omega_A"], bins=bins, alpha=0.6, label=args.label_b, density=True)
axes[0,0].set_xlabel("omega_A (cm$^{-1}$)"); axes[0,0].legend(); axes[0,0].set_title("omega_A distribution")

bins = np.linspace(min(dfA["omega_B"].min(), dfB["omega_B"].min()),
                    max(dfA["omega_B"].max(), dfB["omega_B"].max()), 30)
axes[0,1].hist(dfA["omega_B"], bins=bins, alpha=0.6, label=args.label_a, density=True)
axes[0,1].hist(dfB["omega_B"], bins=bins, alpha=0.6, label=args.label_b, density=True)
axes[0,1].set_xlabel("omega_B (cm$^{-1}$)"); axes[0,1].legend(); axes[0,1].set_title("omega_B distribution")

bins = np.linspace(min(dfA["J_AB_cm1"].min(), dfB["J_AB_cm1"].min()),
                    max(dfA["J_AB_cm1"].max(), dfB["J_AB_cm1"].max()), 30)
axes[1,0].hist(dfA["J_AB_cm1"], bins=bins, alpha=0.6, label=args.label_a, density=True)
axes[1,0].hist(dfB["J_AB_cm1"], bins=bins, alpha=0.6, label=args.label_b, density=True)
axes[1,0].set_xlabel("J_AB (cm$^{-1}$)"); axes[1,0].legend(); axes[1,0].set_title("Coupling distribution")

bins = np.linspace(0, max(dfA["R_AB_Angstrom"].max(), dfB["R_AB_Angstrom"].max()), 30)
axes[1,1].hist(dfA["R_AB_Angstrom"], bins=bins, alpha=0.6, label=args.label_a, density=True)
axes[1,1].hist(dfB["R_AB_Angstrom"], bins=bins, alpha=0.6, label=args.label_b, density=True)
axes[1,1].set_xlabel("R_AB (Angstrom)"); axes[1,1].legend(); axes[1,1].set_title("Distance distribution")

fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"\nSaved: {args.out}")
