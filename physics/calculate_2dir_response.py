"""
calculate_2dir_response.py
===========================
Stage 6 (final) of the amide-I 2D-IR pipeline.

Replaces the previous fixed-width-Gaussian-bump placeholder with a real
third-order sum-over-states (SOS) calculation for a two coupled anharmonic
oscillators (vibrational exciton) system:

  - One-exciton manifold {e-, e+}: eigenstates of the 2x2 Hamiltonian from
    calculate_hamiltonian.py.
  - Two-exciton manifold {f1, f2, f3}: local-mode overtones (AA, BB) and
    combination band (AB), coupled by sqrt(2)*J (standard result for two
    weakly anharmonic coupled oscillators -- see Hamm, Lim & Hochstrasser,
    J. Phys. Chem. B 1998, 102, 6123, the original amide-I coupled
    two-oscillator 2D-IR treatment), built and diagonalized here per
    snapshot using a user-supplied local-mode anharmonicity Delta.
  - Three Feynman pathways: ground-state bleach (GSB), stimulated emission
    (SE), excited-state absorption (ESA, negative sign, shifted along omega3
    by anharmonicity) -- both rephasing and non-rephasing.
  - Secular approximation: no coherence transfer between one-exciton states
    during the waiting time T_wait (diagonal population pathways only for
    SE/ESA). This is a standard, widely used simplification; a full
    treatment would need Redfield/Lindblad population/coherence transfer
    during T_wait, which is out of scope here.
  - Isotropic orientational averaging for all-parallel (ZZZZ) polarization,
    using the actual transition-dipole VECTORS (not just magnitudes) and
    the standard 4th-rank isotropic tensor average
        <(mu1.z)(mu1.z)(mu2.z)(mu2.z)> = (|mu1|^2 |mu2|^2 / 15) * (1 + 2*cos^2(theta12))
  - Static ensemble averaging over your MD/steered-dynamics snapshots for
    the inhomogeneous lineshape (each snapshot is treated as a distinct
    static structure -- appropriate since these come from independent
    frequency calculations, not a continuous trajectory that would let you
    compute a frequency-frequency correlation function / motional
    narrowing). A small homogeneous linewidth (Gamma_homo) is added on top
    to represent fast dephasing not captured by discrete snapshot sampling.

Known limitations (flagged explicitly, not hidden):
  - No coherence transfer / spectral diffusion during T_wait (secular approx).
  - No finite-pulse-bandwidth effects (impulsive limit).
  - Homogeneous lineshape is a simple Lorentzian (frequency-domain), not a
    lineshape derived from a fitted FFCF.
  - Local-mode anharmonicity Delta is a single user-set constant, not
    computed ab initio per residue (the Gaussian calcs here are harmonic).

Usage:
    python calculate_2dir_response.py --hamiltonian amide_hamiltonian_data.csv \
        --wmin 1600 --wmax 1800 --nw 300 --gamma 4.0 --anharmonicity 16.0 \
        --t-wait 0.0 --out-prefix 2DIR
"""

import argparse

import numpy as np
import pandas as pd

SQRT2 = np.sqrt(2.0)


def lorentzian(omega, omega0, gamma, sign=+1):
    """
    Frequency-domain complex Lorentzian: the Fourier transform of a
    one-sided exponentially decaying coherence exp(-i*sign*omega0*t - gamma*t),
    t>0. sign=+1 for a "rephasing-style" -omega1 dependence handled by the
    caller passing omega=-omega1 (see build_pathways), sign is kept explicit
    here only for clarity of convention; gamma is a homogeneous linewidth
    (cm^-1, HWHM in energy units since everything here is in cm^-1).
    """
    return 1j / (omega - sign * omega0 + 1j * gamma)


def orientational_factor(mu1_vec, mu2_vec):
    """
    Isotropic 4th-rank orientational average for all-parallel (ZZZZ)
    polarization, for a pathway where the same dipole direction is probed
    twice at each of two interaction pairs (mu1,mu1) and (mu2,mu2):
        <(mu1.z)^2 (mu2.z)^2> = (|mu1|^2 |mu2|^2 / 15) * (1 + 2*cos^2(theta))
    Standard isotropic-average result (4th-rank tensor average of the
    dipole-dipole-dipole-dipole correlation function); see e.g. Hochstrasser,
    Chem. Phys. 2001, 266, 273, or any 2D-IR/anisotropy reference.
    """
    n1 = np.linalg.norm(mu1_vec)
    n2 = np.linalg.norm(mu2_vec)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_theta = np.dot(mu1_vec, mu2_vec) / (n1 * n2)
    return (n1 ** 2 * n2 ** 2 / 15.0) * (1 + 2 * cos_theta ** 2)


def two_exciton_states(omega_A, omega_B, J, delta_A, delta_B):
    """
    Build and diagonalize the two-exciton (site) Hamiltonian for local modes
    AA, AB (combination), BB:

        H2 = [[2wA-DA,  sqrt2*J,      0    ],
              [sqrt2*J,  wA+wB,   sqrt2*J   ],
              [   0,     sqrt2*J,  2wB-DB   ]]

    Returns (energies[3], eigvecs[3,3]) where eigvecs[:,f] = (c_AA, c_AB, c_BB)
    for two-exciton eigenstate f.
    """
    H2 = np.array([
        [2 * omega_A - delta_A, SQRT2 * J, 0.0],
        [SQRT2 * J, omega_A + omega_B, SQRT2 * J],
        [0.0, SQRT2 * J, 2 * omega_B - delta_B],
    ])
    energies, eigvecs = np.linalg.eigh(H2)
    return energies, eigvecs


def two_exciton_dipole(c_eA, c_eB, mu_A_vec, mu_B_vec, c_fAA, c_fAB, c_fBB):
    """
    Transition dipole vector for a one-exciton state e (coeffs c_eA, c_eB in
    site basis {A,B}) -> two-exciton state f (coeffs c_fAA, c_fAB, c_fBB in
    site basis {AA, AB, BB}), from harmonic ladder-operator algebra applied
    in the site basis (a_A^dagger|1_A> = sqrt(2)|2_A>, etc.):

        mu(e->f) = sqrt(2) c_fAA c_eA mu_A
                 + sqrt(2) c_fBB c_eB mu_B
                 + c_fAB (c_eA mu_B + c_eB mu_A)
    """
    return (SQRT2 * c_fAA * c_eA * mu_A_vec
            + SQRT2 * c_fBB * c_eB * mu_B_vec
            + c_fAB * (c_eA * mu_B_vec + c_eB * mu_A_vec))


def snapshot_response(row, W1, W3, gamma, delta_A, delta_B, t_wait, T1):
    """
    Rephasing and non-rephasing complex response for a single snapshot,
    evaluated on the (W1, W3) meshgrid. Returns (R_rephasing, R_nonrephasing),
    complex arrays same shape as W1/W3.
    """
    omega_A, omega_B, J = row["omega_A"], row["omega_B"], row["J_AB_cm1"]

    mu_A_vec = row["mu_A_Debye"] * np.array([row["muA_hat_x"], row["muA_hat_y"], row["muA_hat_z"]])
    mu_B_vec = row["mu_B_Debye"] * np.array([row["muB_hat_x"], row["muB_hat_y"], row["muB_hat_z"]])

    # one-exciton states
    e_energy = {"minus": row["Omega_minus_cm1"], "plus": row["Omega_plus_cm1"]}
    e_coeff = {
        "minus": (row["c_minus_A"], row["c_minus_B"]),
        "plus": (row["c_plus_A"], row["c_plus_B"]),
    }
    e_dipole = {
        name: c[0] * mu_A_vec + c[1] * mu_B_vec
        for name, c in e_coeff.items()
    }

    # two-exciton states
    f_energies, f_vecs = two_exciton_states(omega_A, omega_B, J, delta_A, delta_B)

    pop_decay = np.exp(-t_wait / T1) if T1 not in (None, np.inf) else 1.0

    R_rephasing = np.zeros_like(W1, dtype=complex)
    R_nonrephasing = np.zeros_like(W1, dtype=complex)

    exciton_names = list(e_energy.keys())

    # ---------------- GSB: independent sum over e, e' ----------------
    for e in exciton_names:
        for ep in exciton_names:
            orient = orientational_factor(e_dipole[e], e_dipole[ep])
            R_rephasing += orient * lorentzian(W1, e_energy[e], gamma, sign=-1) \
                * lorentzian(W3, e_energy[ep], gamma, sign=+1)
            R_nonrephasing += orient * lorentzian(W1, e_energy[e], gamma, sign=+1) \
                * lorentzian(W3, e_energy[ep], gamma, sign=+1)

    # ---------------- SE: diagonal in e (secular approx) ----------------
    for e in exciton_names:
        orient = orientational_factor(e_dipole[e], e_dipole[e])
        R_rephasing += pop_decay * orient * lorentzian(W1, e_energy[e], gamma, sign=-1) \
            * lorentzian(W3, e_energy[e], gamma, sign=+1)
        R_nonrephasing += pop_decay * orient * lorentzian(W1, e_energy[e], gamma, sign=+1) \
            * lorentzian(W3, e_energy[e], gamma, sign=+1)

    # ---------------- ESA: e fixed, sum over two-exciton f, negative sign ----
    for e in exciton_names:
        c_eA, c_eB = e_coeff[e]
        for f_idx in range(3):
            f_energy = f_energies[f_idx]
            c_fAA, c_fAB, c_fBB = f_vecs[:, f_idx]
            mu_ef = two_exciton_dipole(c_eA, c_eB, mu_A_vec, mu_B_vec, c_fAA, c_fAB, c_fBB)
            omega_fe = f_energy - e_energy[e]
            orient = orientational_factor(e_dipole[e], mu_ef)
            R_rephasing -= pop_decay * orient * lorentzian(W1, e_energy[e], gamma, sign=-1) \
                * lorentzian(W3, omega_fe, gamma, sign=+1)
            R_nonrephasing -= pop_decay * orient * lorentzian(W1, e_energy[e], gamma, sign=+1) \
                * lorentzian(W3, omega_fe, gamma, sign=+1)

    return R_rephasing, R_nonrephasing


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hamiltonian", default="amide_hamiltonian_data.csv")
    p.add_argument("--wmin", type=float, default=1600.0)
    p.add_argument("--wmax", type=float, default=1800.0)
    p.add_argument("--nw", type=int, default=300)
    p.add_argument("--gamma", type=float, default=4.0,
                    help="Homogeneous linewidth (cm^-1, Lorentzian HWHM in "
                         "energy units) added on top of the snapshot-to-"
                         "snapshot (inhomogeneous) frequency distribution.")
    p.add_argument("--anharmonicity", type=float, default=16.0,
                    help="Local-mode diagonal anharmonicity Delta (cm^-1), "
                         "applied to both oscillators. ~14-16 cm^-1 is a "
                         "commonly cited value for the amide-I mode; "
                         "override if you have a system-specific estimate.")
    p.add_argument("--t-wait", type=float, default=0.0,
                    help="Waiting time T2 (in the same time units as --t1-relaxation).")
    p.add_argument("--t1-relaxation", type=float, default=np.inf,
                    help="Phenomenological vibrational population lifetime T1 "
                         "for SE/ESA pathways; population decay is "
                         "exp(-t_wait/T1). Default: no decay (T1=inf).")
    p.add_argument("--out-prefix", default="2DIR")
    args = p.parse_args()

    df = pd.read_csv(args.hamiltonian)
    print(f"Averaging over {len(df)} snapshots (static ensemble average)")

    w1 = np.linspace(args.wmin, args.wmax, args.nw)
    w3 = np.linspace(args.wmin, args.wmax, args.nw)
    W1, W3 = np.meshgrid(w1, w3)

    R_rephasing_total = np.zeros_like(W1, dtype=complex)
    R_nonrephasing_total = np.zeros_like(W1, dtype=complex)

    for _, row in df.iterrows():
        R_r, R_nr = snapshot_response(
            row, W1, W3, args.gamma, args.anharmonicity, args.anharmonicity,
            args.t_wait, args.t1_relaxation,
        )
        R_rephasing_total += R_r
        R_nonrephasing_total += R_nr

    R_rephasing_total /= len(df)
    R_nonrephasing_total /= len(df)

    # Purely absorptive 2D spectrum: standard combination of rephasing +
    # non-rephasing (the -omega1 sign convention baked into the rephasing
    # Lorentzians above means both are evaluated on the SAME physical,
    # positive omega1 axis and simply added -- see e.g. Hamm & Zanni,
    # "Concepts and Methods of 2D Infrared Spectroscopy", Ch. 4.
    absorptive = np.real(R_rephasing_total + R_nonrephasing_total)

    norm = np.max(np.abs(absorptive))
    absorptive_norm = absorptive / norm if norm > 0 else absorptive

    np.save(f"{args.out_prefix}_rephasing.npy", R_rephasing_total)
    np.save(f"{args.out_prefix}_nonrephasing.npy", R_nonrephasing_total)
    np.save(f"{args.out_prefix}_absorptive.npy", absorptive_norm)
    np.save(f"{args.out_prefix}_w1.npy", w1)
    np.save(f"{args.out_prefix}_w3.npy", w3)
    np.savetxt(f"{args.out_prefix}_absorptive.dat", absorptive_norm)

    print("\n2D-IR response calculation completed.")
    print("Grid size:", absorptive.shape)
    print("Frequency range:", args.wmin, "to", args.wmax, "cm^-1")
    print("Homogeneous linewidth (Gamma):", args.gamma, "cm^-1")
    print("Anharmonicity (Delta):", args.anharmonicity, "cm^-1")
    print("Waiting time:", args.t_wait)
    print("Maximum (normalized):", absorptive_norm.max())
    print("Minimum (normalized):", absorptive_norm.min())
    print(f"\nSaved: {args.out_prefix}_rephasing.npy, {args.out_prefix}_nonrephasing.npy, "
          f"{args.out_prefix}_absorptive.npy(.dat), {args.out_prefix}_w1.npy, {args.out_prefix}_w3.npy")

    # Optional quick-look plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        levels = np.linspace(-1, 1, 21)
        cf = ax.contourf(W1, W3, absorptive_norm, levels=levels, cmap="RdBu_r")
        ax.contour(W1, W3, absorptive_norm, levels=levels, colors="k", linewidths=0.3)
        ax.plot([args.wmin, args.wmax], [args.wmin, args.wmax], "k--", linewidth=0.8)
        ax.set_xlabel(r"$\omega_1$ (cm$^{-1}$)")
        ax.set_ylabel(r"$\omega_3$ (cm$^{-1}$)")
        ax.set_title(f"Absorptive 2D-IR spectrum, $T_w$={args.t_wait}")
        fig.colorbar(cf, ax=ax, label="Normalized signal")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_absorptive.png", dpi=200)
        print(f"Saved: {args.out_prefix}_absorptive.png")
    except ImportError:
        print("matplotlib not available -- skipped PNG preview.")


if __name__ == "__main__":
    main()
