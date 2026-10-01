"""
calculate_transition_dipoles.py
================================
Stage 5 of the amide-I 2D-IR pipeline.

Takes the per-snapshot Hamiltonian data (which already carries the
physically calibrated absolute transition dipoles mu_A_Debye/mu_B_Debye
from calculate_coupling.py) and adds derived quantities used for
plotting/analysis and as the final input table for the 2D-IR response
calculation:

  - mu_A_norm/mu_B_norm : dipoles normalized to this dataset's max
    (convenience for plotting relative peak intensities only -- the
    physically meaningful absolute values are mu_A_Debye/mu_B_Debye,
    already present from the coupling stage; use those for anything
    that should be comparable across different datasets/residue pairs)
  - strength_A/B         : normalized oscillator strength (mu_norm^2)
  - delta_omega, mean_frequency
  - J_over_detuning       : |J| / |delta_omega|, a rough diagnostic of
    how strongly mixed (exciton-like, near degenerate) vs. how
    weakly-perturbed (local-mode-like) each snapshot's pair is

Usage:
    python calculate_transition_dipoles.py --hamiltonian amide_hamiltonian_data.csv \
        --out amide_2dir_parameters.csv
"""

import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hamiltonian", default="amide_hamiltonian_data.csv")
    p.add_argument("--out", default="amide_2dir_parameters.csv")
    args = p.parse_args()

    df = pd.read_csv(args.hamiltonian)

    df["mu_A_norm"] = df["mu_A_Debye"] / df["mu_A_Debye"].max()
    df["mu_B_norm"] = df["mu_B_Debye"] / df["mu_B_Debye"].max()

    df["strength_A"] = df["mu_A_norm"] ** 2
    df["strength_B"] = df["mu_B_norm"] ** 2

    df["delta_omega"] = df["omega_A"] - df["omega_B"]
    df["mean_frequency"] = (df["omega_A"] + df["omega_B"]) / 2.0

    df["J_over_detuning"] = (
        df["J_AB_cm1"].abs() / np.maximum(df["delta_omega"].abs(), 1e-12)
    )

    df.to_csv(args.out, index=False)

    print("\nSaved:", args.out)
    print("\nAbsolute transition dipole statistics (Debye):")
    print(df[["mu_A_Debye", "mu_B_Debye"]].describe())
    print("\nMean frequency:", df["mean_frequency"].mean())
    print("Frequency standard deviation:", df["mean_frequency"].std())
    print("\nMean |J/delta omega|:", df["J_over_detuning"].mean())
    print("Maximum |J/delta omega|:", df["J_over_detuning"].max())
    print("\nCalculation completed.")


if __name__ == "__main__":
    main()
