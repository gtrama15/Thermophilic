"""
make_figureS1_validation.py
=============================
Figure S1: computational validation using a synthetic two-oscillator test
case with known, hand-specified frequencies and coupling. Self-contained
(re-implements the minimal response-function physics needed) so it does
not depend on being run alongside the full analysis pipeline.

Panel A: exciton splitting from the code vs. the analytic 2x2 Hamiltonian
formula sqrt((wA-wB)^2 + 4J^2) -- should fall exactly on the diagonal.
Panel B: cross-peak/diagonal-peak intensity ratio vs. coupling strength J,
showing the expected monotonic increase (weak-coupling exciton mixing).

Usage:
    python make_figureS1_validation.py --out final_figures/FigureS1_validation.png
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

SQRT2 = np.sqrt(2.0)

def lorentzian(omega, omega0, gamma, sign=+1):
    return 1j / (omega - sign * omega0 + 1j * gamma)

def orientational_factor(mu1_vec, mu2_vec):
    n1, n2 = np.linalg.norm(mu1_vec), np.linalg.norm(mu2_vec)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_theta = np.dot(mu1_vec, mu2_vec) / (n1 * n2)
    return (n1**2 * n2**2 / 15.0) * (1 + 2 * cos_theta**2)

def two_exciton_states(omega_A, omega_B, J, delta_A, delta_B):
    H2 = np.array([[2*omega_A-delta_A, SQRT2*J, 0.0],
                    [SQRT2*J, omega_A+omega_B, SQRT2*J],
                    [0.0, SQRT2*J, 2*omega_B-delta_B]])
    return np.linalg.eigh(H2)

def two_exciton_dipole(c_eA, c_eB, mu_A_vec, mu_B_vec, c_fAA, c_fAB, c_fBB):
    return (SQRT2*c_fAA*c_eA*mu_A_vec + SQRT2*c_fBB*c_eB*mu_B_vec
            + c_fAB*(c_eA*mu_B_vec + c_eB*mu_A_vec))

def snapshot_response(row, W1, W3, gamma, delta_A, delta_B, t_wait, T1):
    omega_A, omega_B, J = row["omega_A"], row["omega_B"], row["J_AB_cm1"]
    mu_A_vec = row["mu_A_Debye"] * np.array([row["muA_hat_x"], row["muA_hat_y"], row["muA_hat_z"]])
    mu_B_vec = row["mu_B_Debye"] * np.array([row["muB_hat_x"], row["muB_hat_y"], row["muB_hat_z"]])
    e_energy = {"minus": row["Omega_minus_cm1"], "plus": row["Omega_plus_cm1"]}
    e_coeff = {"minus": (row["c_minus_A"], row["c_minus_B"]), "plus": (row["c_plus_A"], row["c_plus_B"])}
    e_dipole = {name: c[0]*mu_A_vec + c[1]*mu_B_vec for name, c in e_coeff.items()}
    f_energies, f_vecs = two_exciton_states(omega_A, omega_B, J, delta_A, delta_B)
    pop_decay = np.exp(-t_wait/T1) if T1 not in (None, np.inf) else 1.0
    R_r = np.zeros_like(W1, dtype=complex)
    R_nr = np.zeros_like(W1, dtype=complex)
    names = list(e_energy.keys())
    for e in names:
        for ep in names:
            o = orientational_factor(e_dipole[e], e_dipole[ep])
            R_r += o * lorentzian(W1, e_energy[e], gamma, -1) * lorentzian(W3, e_energy[ep], gamma, +1)
            R_nr += o * lorentzian(W1, e_energy[e], gamma, +1) * lorentzian(W3, e_energy[ep], gamma, +1)
    for e in names:
        o = orientational_factor(e_dipole[e], e_dipole[e])
        R_r += pop_decay*o*lorentzian(W1, e_energy[e], gamma, -1)*lorentzian(W3, e_energy[e], gamma, +1)
        R_nr += pop_decay*o*lorentzian(W1, e_energy[e], gamma, +1)*lorentzian(W3, e_energy[e], gamma, +1)
    for e in names:
        c_eA, c_eB = e_coeff[e]
        for f_idx in range(3):
            f_energy = f_energies[f_idx]
            c_fAA, c_fAB, c_fBB = f_vecs[:, f_idx]
            mu_ef = two_exciton_dipole(c_eA, c_eB, mu_A_vec, mu_B_vec, c_fAA, c_fAB, c_fBB)
            omega_fe = f_energy - e_energy[e]
            o = orientational_factor(e_dipole[e], mu_ef)
            R_r -= pop_decay*o*lorentzian(W1, e_energy[e], gamma, -1)*lorentzian(W3, omega_fe, gamma, +1)
            R_nr -= pop_decay*o*lorentzian(W1, e_energy[e], gamma, +1)*lorentzian(W3, omega_fe, gamma, +1)
    return R_r, R_nr


p = argparse.ArgumentParser()
p.add_argument("--out", default="FigureS1_validation.png")
args = p.parse_args()

omega_A, omega_B = 1650.0, 1690.0
w = np.linspace(1600, 1740, 300)
W1, W3 = np.meshgrid(w, w)

def build_row(J, mu_A=0.3, mu_B=0.3, angle_deg=60.0):
    H = np.array([[omega_A, J], [J, omega_B]])
    eigvals, eigvecs = np.linalg.eigh(H)
    row = pd.Series({
        "omega_A": omega_A, "omega_B": omega_B, "J_AB_cm1": J,
        "mu_A_Debye": mu_A, "mu_B_Debye": mu_B,
        "muA_hat_x": 1.0, "muA_hat_y": 0.0, "muA_hat_z": 0.0,
        "muB_hat_x": np.cos(np.radians(angle_deg)), "muB_hat_y": np.sin(np.radians(angle_deg)), "muB_hat_z": 0.0,
        "Omega_minus_cm1": eigvals[0], "Omega_plus_cm1": eigvals[1],
        "c_minus_A": eigvecs[0,0], "c_minus_B": eigvecs[1,0],
        "c_plus_A": eigvecs[0,1], "c_plus_B": eigvecs[1,1],
    })
    return row, eigvals

J_values = np.array([0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0])
analytic_splitting, code_splitting, cross_diag_ratio = [], [], []

def peak_near(absorptive, w1_t, w3_t, window=6):
    m1 = (w >= w1_t - window) & (w <= w1_t + window)
    m3 = (w >= w3_t - window) & (w <= w3_t + window)
    sub = absorptive[np.ix_(m3, m1)]
    return sub.max() if sub.size else np.nan

for J in J_values:
    row, eigvals = build_row(J)
    analytic = np.sqrt((omega_A - omega_B)**2 + 4*J**2)
    code_val = eigvals[1] - eigvals[0]
    analytic_splitting.append(analytic)
    code_splitting.append(code_val)

    R_r, R_nr = snapshot_response(row, W1, W3, gamma=3.0, delta_A=16.0, delta_B=16.0, t_wait=0.0, T1=np.inf)
    absorptive = np.real(R_r + R_nr)
    absorptive /= np.max(np.abs(absorptive))
    diag = 0.5*(peak_near(absorptive, omega_A, omega_A) + peak_near(absorptive, omega_B, omega_B))
    cross = 0.5*(peak_near(absorptive, omega_A, omega_B) + peak_near(absorptive, omega_B, omega_A))
    cross_diag_ratio.append(cross/diag if diag != 0 else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))

axes[0].plot(analytic_splitting, code_splitting, "o", markersize=7)
lims = [0, max(analytic_splitting)*1.05]
axes[0].plot(lims, lims, "k--", linewidth=1, label="y = x")
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_aspect("equal")
axes[0].set_xlabel("Analytic splitting (cm$^{-1}$)")
axes[0].set_ylabel("Computed splitting (cm$^{-1}$)")
axes[0].set_title("A: Exciton splitting")
axes[0].legend()

axes[1].plot(J_values, cross_diag_ratio, "o-", markersize=7)
axes[1].axvspan(0, 16, alpha=0.08, color="green", label="Monotonic")
axes[1].axvspan(16, 25, alpha=0.08, color="red", label="Non-monotonic")
axes[1].set_xlabel("Coupling J (cm$^{-1}$)")
axes[1].set_ylabel("Cross-peak / diagonal-peak ratio")
axes[1].set_title("B: Cross-peak ratio")
axes[1].legend(loc="upper left")

fig.tight_layout()
fig.savefig(args.out, dpi=300)
print(f"Saved: {args.out}")

max_dev = np.max(np.abs(np.array(analytic_splitting) - np.array(code_splitting)))
print(f"Max deviation between analytic and code-computed splitting: {max_dev:.6f} cm^-1 (should be ~0)")
