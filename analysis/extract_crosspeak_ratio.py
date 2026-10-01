"""
extract_crosspeak_ratio.py
============================
Quantitative cross-peak/diagonal-peak intensity ratio from a computed 2D-IR
spectrum -- the defensible, number-based comparison between two ensembles'
spectra (e.g. Close vs Far), since the coupling's effect on the OVERALL
spectral shape can be subtle compared to the dominant diagonal-frequency
scatter (see validate_response.py for the same diagnostic on synthetic data).

Usage:
    python extract_crosspeak_ratio.py --prefix 2DIR_2_27_close --hamiltonian ham_2_27_close.csv
    python extract_crosspeak_ratio.py --prefix 2DIR_2_27_far --hamiltonian ham_2_27_far.csv
"""
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--prefix", required=True, help="Prefix used in calculate_2dir_response.py --out-prefix")
p.add_argument("--hamiltonian", required=True, help="The Hamiltonian CSV used to generate this spectrum")
p.add_argument("--window", type=float, default=6.0, help="Search window (cm^-1) around each peak position")
args = p.parse_args()

absorptive = np.load(f"{args.prefix}_absorptive.npy")
w1 = np.load(f"{args.prefix}_w1.npy")
w3 = np.load(f"{args.prefix}_w3.npy")

df = pd.read_csv(args.hamiltonian)
mean_omega_A = df["omega_A"].mean()
mean_omega_B = df["omega_B"].mean()
print(f"Ensemble mean omega_A={mean_omega_A:.2f}, omega_B={mean_omega_B:.2f} cm^-1")

def peak_near(w1_target, w3_target, window):
    m1 = (w1 >= w1_target - window) & (w1 <= w1_target + window)
    m3 = (w3 >= w3_target - window) & (w3 <= w3_target + window)
    sub = absorptive[np.ix_(m3, m1)]
    return sub.max() if sub.size else np.nan


def peak_near_local_max(w1_target, w3_target, window):
    """
    More robust version: instead of assuming the peak sits exactly at the
    ensemble mean frequency, search a window around it and return the
    actual local extremum (whichever has larger |value|, to correctly
    find negative ESA-dominated regions too) along with WHERE it was found
    -- useful for diagnosing cases (like closely-spaced omega_A/omega_B)
    where GSB/SE and ESA features overlap and the simple fixed-point probe
    can give a misleading sign/magnitude.
    """
    m1 = (w1 >= w1_target - window) & (w1 <= w1_target + window)
    m3 = (w3 >= w3_target - window) & (w3 <= w3_target + window)
    sub = absorptive[np.ix_(m3, m1)]
    if sub.size == 0:
        return np.nan, np.nan, np.nan
    idx = np.unravel_index(np.argmax(np.abs(sub)), sub.shape)
    w1_sub, w3_sub = w1[m1], w3[m3]
    return sub[idx], w1_sub[idx[1]], w3_sub[idx[0]]

diag_A = peak_near(mean_omega_A, mean_omega_A, args.window)
diag_B = peak_near(mean_omega_B, mean_omega_B, args.window)
cross_AB = peak_near(mean_omega_A, mean_omega_B, args.window)
cross_BA = peak_near(mean_omega_B, mean_omega_A, args.window)

diag_avg = 0.5 * (diag_A + diag_B)
cross_avg = 0.5 * (cross_AB + cross_BA)
ratio = cross_avg / diag_avg if diag_avg != 0 else np.nan

print(f"\nDiagonal peak near (A,A): {diag_A:.4f}")
print(f"Diagonal peak near (B,B): {diag_B:.4f}")
print(f"Cross peak near (A,B): {cross_AB:.4f}")
print(f"Cross peak near (B,A): {cross_BA:.4f}")
print(f"\nCross/diagonal ratio: {ratio:.4f}")

print(f"\n--- Local-maximum version (more robust when omega_A/omega_B are close together) ---")
val_A, w1_A, w3_A = peak_near_local_max(mean_omega_A, mean_omega_A, args.window)
val_B, w1_B, w3_B = peak_near_local_max(mean_omega_B, mean_omega_B, args.window)
val_AB, w1_AB, w3_AB = peak_near_local_max(mean_omega_A, mean_omega_B, args.window)
val_BA, w1_BA, w3_BA = peak_near_local_max(mean_omega_B, mean_omega_A, args.window)
print(f"Diagonal (A,A): value={val_A:.4f} at (w1={w1_A:.1f}, w3={w3_A:.1f})")
print(f"Diagonal (B,B): value={val_B:.4f} at (w1={w1_B:.1f}, w3={w3_B:.1f})")
print(f"Cross (A,B): value={val_AB:.4f} at (w1={w1_AB:.1f}, w3={w3_AB:.1f})")
print(f"Cross (B,A): value={val_BA:.4f} at (w1={w1_BA:.1f}, w3={w3_BA:.1f})")
diag_avg2 = 0.5 * (abs(val_A) + abs(val_B))
cross_avg2 = 0.5 * (abs(val_AB) + abs(val_BA))
ratio2 = cross_avg2 / diag_avg2 if diag_avg2 != 0 else np.nan
print(f"Cross/diagonal ratio (local-max, magnitude-based): {ratio2:.4f}")

print("\n(Compare this ratio directly between your Close and Far spectra -- "
      "a meaningfully larger ratio for Close is the quantitative signature "
      "of real coupling, independent of how similar the spectra look visually.)")
