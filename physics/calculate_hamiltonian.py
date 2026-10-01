"""
calculate_hamiltonian.py
=========================
Stage 4 of the amide-I 2D-IR pipeline.

Builds and diagonalizes the 2x2 vibrational exciton Hamiltonian

    H = [[omega_A,  J_AB],
         [J_AB,      omega_B]]

per snapshot. Carries all upstream columns (mu_A/B_Debye, kappa,
R_AB, IR_A/B, ...) through to the output so downstream stages don't need
to re-merge with earlier CSVs.

Usage:
    python calculate_hamiltonian.py --coupling amide_coupling_data.csv \
        --out amide_hamiltonian_data.csv
"""

import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coupling", default="amide_coupling_data.csv")
    p.add_argument("--out", default="amide_hamiltonian_data.csv")
    args = p.parse_args()

    df = pd.read_csv(args.coupling)

    Omega_minus = np.empty(len(df))
    Omega_plus = np.empty(len(df))
    # One-exciton eigenvector coefficients in the site basis {A, B}:
    # lower eigenstate e- = c_minus_A * |A> + c_minus_B * |B>, likewise e+.
    # Needed downstream to build exciton-basis transition dipole VECTORS
    # (magnitude alone isn't enough for the 2D-IR orientational averaging).
    c_minus_A = np.empty(len(df))
    c_minus_B = np.empty(len(df))
    c_plus_A = np.empty(len(df))
    c_plus_B = np.empty(len(df))

    for idx, row in enumerate(df.itertuples()):
        H = np.array([[row.omega_A, row.J_AB_cm1],
                      [row.J_AB_cm1, row.omega_B]])
        eigenvalues, eigenvectors = np.linalg.eigh(H)  # columns = eigenvectors, ascending order
        Omega_minus[idx] = eigenvalues[0]
        Omega_plus[idx] = eigenvalues[1]
        c_minus_A[idx], c_minus_B[idx] = eigenvectors[:, 0]
        c_plus_A[idx], c_plus_B[idx] = eigenvectors[:, 1]

    output = df.copy()
    output["Omega_minus_cm1"] = Omega_minus
    output["Omega_plus_cm1"] = Omega_plus
    output["exciton_splitting_cm1"] = Omega_plus - Omega_minus
    output["c_minus_A"] = c_minus_A
    output["c_minus_B"] = c_minus_B
    output["c_plus_A"] = c_plus_A
    output["c_plus_B"] = c_plus_B

    output.to_csv(args.out, index=False)

    print("\nSaved:", args.out)
    print("\nExciton frequency statistics:")
    print(output[["Omega_minus_cm1", "Omega_plus_cm1", "exciton_splitting_cm1"]].describe())
    print("\nFirst snapshots:")
    print(output.head(10).to_string(index=False))
    print("\nCalculation completed.")


if __name__ == "__main__":
    main()
