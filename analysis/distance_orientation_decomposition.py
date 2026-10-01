"""
distance_orientation_decomposition.py (CORRECTED)
=====================================================
Decomposes phosphorylation-state and temperature differences in J_AB into
distance-driven vs. orientation-driven contributions, using the identity

    J_AB = TDC_PREFACTOR_A * kappa * (TDC_DIPOLE_MAGNITUDE^2) / R_AB^3

NOTE ON A PREVIOUS BUG (fixed here): an earlier version of this script
incorrectly used the ensemble-mean of mu_A_Debye * mu_B_Debye (the
frequency-dependent transition DIPOLE MOMENT, used only for the separate
2D-IR response-function calculation) in place of the correct quantity --
the FIXED TDC dipole-derivative magnitude (2.73 D/(Angstrom*sqrt(u)),
Torii-Tasumi parameter, identical for every snapshot by construction).
This caused the counterfactual "actual" J_AB reconstruction to be off by
a factor of ~(0.28/2.73)^2 ~= 1/94 relative to the real ensemble mean.
Since this dipole magnitude is fixed by the model, it correctly drops out
of the distance-vs-orientation decomposition entirely (it cannot be a
condition-to-condition varying quantity in this model).

Requires coupling CSVs that include a "kappa" column.

Usage:
    python distance_orientation_decomposition.py \
        --a ensemble_A_with_kappa.csv --b ensemble_B_with_kappa.csv \
        --label-a "1QMP, 300K" --label-b "1DZ3, 300K" \
        --out TableS4_distance_orientation_decomposition.csv
"""
import argparse
import numpy as np
import pandas as pd

TDC_PREFACTOR_A = 0.1 * 848619.0 / 1650.0
TDC_DIPOLE_MAGNITUDE = 2.73  # D/(Angstrom*sqrt(u)), FIXED by the model -- not an ensemble-varying quantity
MU_SQ = TDC_DIPOLE_MAGNITUDE ** 2

p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--label-a", default="A")
p.add_argument("--label-b", default="B")
p.add_argument("--out", default="TableS4_distance_orientation_decomposition.csv")
args = p.parse_args()

dfA = pd.read_csv(args.a)
dfB = pd.read_csv(args.b)

for label, path, df in [(args.label_a, args.a, dfA), (args.label_b, args.b, dfB)]:
    if "kappa" not in df.columns:
        raise SystemExit(
            f"ERROR: {label} ({path}) has no 'kappa' column. "
            f"Apply the patch in kappa_patch_instructions.py and re-run "
            f"compute_map_coupling_parallel.py on this ensemble first."
        )

rows = []
for label, df in [(args.label_a, dfA), (args.label_b, dfB)]:
    r_mean, k_mean, j_mean = df["R_AB_Angstrom"].mean(), df["kappa"].mean(), df["J_AB_cm1"].mean()
    r_corr = df["R_AB_Angstrom"].corr(df["J_AB_cm1"])
    k_corr = df["kappa"].corr(df["J_AB_cm1"])
    rows.append({
        "condition": label, "n": len(df),
        "R_AB_mean": round(r_mean, 3), "kappa_mean": round(k_mean, 3),
        "J_AB_mean (actual, real data)": round(j_mean, 4),
        "corr(J_AB, R_AB)": round(r_corr, 3), "corr(J_AB, kappa)": round(k_corr, 3),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

# --- Counterfactual decomposition (mean-field), using the FIXED dipole
#     magnitude -- only R and kappa vary between conditions in this model ---
meanA = {"R": dfA["R_AB_Angstrom"].mean(), "k": dfA["kappa"].mean()}
meanB = {"R": dfB["R_AB_Angstrom"].mean(), "k": dfB["kappa"].mean()}

def J_from_means(R, k):
    return TDC_PREFACTOR_A * k * MU_SQ / R**3

J_A_reconstructed = J_from_means(meanA["R"], meanA["k"])
J_B_reconstructed = J_from_means(meanB["R"], meanB["k"])
J_B_with_A_distance = J_from_means(meanA["R"], meanB["k"])    # B's orientation, A's distance
J_B_with_A_orientation = J_from_means(meanB["R"], meanA["k"])  # B's distance, A's orientation

total_diff = J_B_reconstructed - J_A_reconstructed
distance_contribution = J_B_with_A_orientation - J_A_reconstructed
orientation_contribution = J_B_with_A_distance - J_A_reconstructed

print("\n" + "="*70)
print("COUNTERFACTUAL DECOMPOSITION (mean-field approximation)")
print("="*70)
print(f"J_AB, {args.label_a}: actual (real data) = {dfA['J_AB_cm1'].mean():.4f} cm^-1, "
      f"reconstructed (mean R,kappa) = {J_A_reconstructed:.4f} cm^-1")
print(f"J_AB, {args.label_b}: actual (real data) = {dfB['J_AB_cm1'].mean():.4f} cm^-1, "
      f"reconstructed (mean R,kappa) = {J_B_reconstructed:.4f} cm^-1")
print("(reconstructed vs. actual should now be close -- residual difference is the")
print(" expected mean-field approximation error, since E[1/R^3] != 1/E[R]^3 exactly)")
print(f"\nTotal reconstructed difference ({args.label_b} - {args.label_a}): {total_diff:.4f} cm^-1")
if abs(total_diff) > 1e-10:
    print(f"  Contribution from distance change alone (orientation fixed): {distance_contribution:.4f} cm^-1 "
          f"({100*distance_contribution/total_diff:.0f}% of total)")
    print(f"  Contribution from orientation change alone (distance fixed): {orientation_contribution:.4f} cm^-1 "
          f"({100*orientation_contribution/total_diff:.0f}% of total)")

summary.to_csv(args.out, index=False)
print(f"\nSaved: {args.out}")
