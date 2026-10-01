"""
map_physics.py
===============
Implements the Wang, Middleton, Zanni & Skinner (2011, JPCB 115, 3713)
backbone amide-I frequency map and the Torii-Tasumi TDC coupling scheme
exactly as specified in that paper's Eqs. 11 and 14, applied to raw MD
snapshots + GROMACS partial charges (rather than the paper's own use of
in-house MD electric-field output).

All physical constants/parameters below (map coefficients, TDC dipole
magnitude/orientation, coupling prefactor) are taken directly from the
paper, not re-derived -- see docstrings for the exact equation numbers.
"""

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# ------------------------------------------------------------------
# Backbone frequency map (Wang et al. 2011, Eq. 11)
#   omega_i = 1684 + 7729*E_Ci - 3576*E_Ni   (cm^-1; E in atomic units)
# ------------------------------------------------------------------
MAP_OMEGA0 = 1684.0
MAP_A_EC = 7729.0
MAP_B_EN = -3576.0

# ------------------------------------------------------------------
# TDC coupling parameters (Wang et al. 2011, Eq. 14, citing Torii & Tasumi)
# ------------------------------------------------------------------
TDC_DIPOLE_MAGNITUDE = 2.73       # D * Angstrom^-1 * u^-1/2
TDC_DIPOLE_ANGLE_DEG = 10.0       # degrees off the C=O bond, toward N
TDC_ORIGIN_A = 0.665              # coefficient on n_CO
TDC_ORIGIN_B = 0.258              # coefficient on n_CN
TDC_PREFACTOR_A = 0.1 * 848619.0 / 1650.0  # gives coupling directly in cm^-1


def minimum_image(displacement, box):
    """Apply orthorhombic minimum-image convention. displacement, box: (...,3)."""
    return displacement - box * np.round(displacement / box)


def electric_field_at_point(field_point, source_coords, source_charges, box,
                             cutoff_angstrom=20.0, exclude_mask=None):
    """
    Electric field (atomic units) at field_point (Angstrom) due to all
    source_coords/source_charges within cutoff_angstrom, using minimum-image
    PBC. exclude_mask: boolean array, True = exclude that source atom
    (e.g. the chromophore's own C,O,N,H atoms, per Wang et al. Sec. III.A).

    Returns the field VECTOR (atomic units, i.e. e/bohr^2).
    """
    disp = source_coords - field_point  # (N,3), Angstrom, points TO source
    disp = minimum_image(disp, box)
    r2 = np.sum(disp ** 2, axis=1)

    mask = (r2 < cutoff_angstrom ** 2) & (r2 > 1e-6)
    if exclude_mask is not None:
        mask &= ~exclude_mask

    disp_bohr = disp[mask] * ANGSTROM_TO_BOHR
    r2_bohr = np.sum(disp_bohr ** 2, axis=1)
    r_bohr = np.sqrt(r2_bohr)
    q = source_charges[mask]

    # Field at field_point due to charge q at disp away: E = q * r_hat / r^2
    # disp points FROM field_point TO source, so the field contribution from
    # a positive charge points back toward field_point along -disp_hat;
    # standard formula: E(P) = sum q_j (P - r_j) / |P - r_j|^3 = sum q_j (-disp_bohr) / r_bohr^3
    contributions = -q[:, None] * disp_bohr / (r_bohr ** 3)[:, None]
    return contributions.sum(axis=0)


def compute_backbone_frequency(C_coord, O_coord, N_coord, H_coord,
                                all_coords, all_charges, box,
                                exclude_indices_C, exclude_indices_N,
                                cutoff_angstrom=20.0):
    """
    Compute the backbone amide-I frequency for one chromophore via the
    Wang et al. 2011 map (Eq. 11): omega = 1684 + 7729*E_C - 3576*E_N,
    where E_C, E_N are field components along the C=O bond direction at
    the C and N atoms.

    exclude_indices_C, exclude_indices_N: boolean arrays (len = n_atoms),
    True = exclude that source atom from the field at C (resp. N). These
    should be built from the topology's actual bonded-exclusion list
    (1-2, 1-3 neighbors, matching nrexcl), NOT just the 4 backbone atoms --
    covalently close atoms (e.g. C-alpha) produce unphysically large raw
    Coulomb fields if not excluded, exactly as real force fields exclude
    them from nonbonded interactions. See topology_charges.build_exclusion_sets.
    """
    n_co = (O_coord - C_coord)
    n_co = n_co / np.linalg.norm(n_co)

    E_C_vec = electric_field_at_point(C_coord, all_coords, all_charges, box,
                                       cutoff_angstrom, exclude_indices_C)
    E_N_vec = electric_field_at_point(N_coord, all_coords, all_charges, box,
                                       cutoff_angstrom, exclude_indices_N)

    E_C = np.dot(E_C_vec, n_co)
    E_N = np.dot(E_N_vec, n_co)

    omega = MAP_OMEGA0 + MAP_A_EC * E_C + MAP_B_EN * E_N
    return omega, E_C, E_N


def transition_dipole_vector(C_coord, O_coord, N_coord):
    """
    TDC transition dipole vector per Wang et al. 2011 Eq. 14 / Torii-Tasumi:
    magnitude 2.73 D/(Angstrom*sqrt(u)), oriented 10 degrees off the C=O
    bond toward N, origin at r_C + 0.665*n_CO + 0.258*n_CN.

    Returns (origin (Angstrom), dipole_vector (D/(Angstrom*sqrt(u)))).
    """
    n_co = (O_coord - C_coord)
    n_co = n_co / np.linalg.norm(n_co)
    n_cn = (N_coord - C_coord)
    n_cn = n_cn / np.linalg.norm(n_cn)

    origin = C_coord + TDC_ORIGIN_A * n_co + TDC_ORIGIN_B * n_cn

    # Rotate n_co by TDC_DIPOLE_ANGLE_DEG toward n_cn, within the C-O-N plane.
    # Build an orthonormal in-plane basis: e1 = n_co, e2 = component of n_cn
    # perpendicular to n_co.
    e1 = n_co
    e2 = n_cn - np.dot(n_cn, e1) * e1
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-8:
        e2 = np.array([0.0, 0.0, 0.0])
    else:
        e2 = e2 / e2_norm

    theta = np.radians(TDC_DIPOLE_ANGLE_DEG)
    direction = np.cos(theta) * e1 + np.sin(theta) * e2

    dipole_vector = TDC_DIPOLE_MAGNITUDE * direction
    return origin, dipole_vector


def tdc_coupling(origin_A, mu_A, origin_B, mu_B, box):
    """
    TDC coupling constant (cm^-1) per Wang et al. 2011 Eq. 14:
      omega_ij = A * { (mu_i.mu_j)/r^3 - 3*(mu_i.rhat)(mu_j.rhat)/r^3 }
    with A = 0.1*848619/1650, dielectric constant epsilon=1, and mu in
    D/(Angstrom*sqrt(u)) (so this directly gives cm^-1, no separate unit
    conversion needed -- the mass-weighted-derivative-to-transition-dipole
    conversion is already folded into the calibrated prefactor A).
    """
    r_vec = origin_B - origin_A
    r_vec = minimum_image(r_vec, box)
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r

    dipole_dot = np.dot(mu_A, mu_B)
    orientation_term = np.dot(mu_A, r_hat) * np.dot(mu_B, r_hat)

    mu_A_hat = mu_A / np.linalg.norm(mu_A)
    mu_B_hat = mu_B / np.linalg.norm(mu_B)
    kappa = np.dot(mu_A_hat, mu_B_hat) - 3 * np.dot(mu_A_hat, r_hat) * np.dot(mu_B_hat, r_hat)

    J = TDC_PREFACTOR_A * (dipole_dot - 3 * orientation_term) / r ** 3
    return J, kappa


def transition_dipole_debye(mu_derivative_vector, omega_cm1):
    """
    Convert the TDC dipole-DERIVATIVE vector (D/(Angstrom*sqrt(u)), as used
    directly in tdc_coupling above) into an actual transition dipole MOMENT
    in Debye, using the standard harmonic mass-weighted-normal-coordinate
    relation (see amide2dir_constants.py: <1|Q|0> = sqrt(16.8576/omega)).

    NOTE: this conversion is NOT part of the original Wang et al. 2011
    paper (which only computes linear IR absorption via the F(t) propagator
    and never needs an explicit Debye-unit dipole). It is needed here
    specifically to feed this map's output into the separately-built
    third-order 2D-IR response function machinery (calculate_2dir_response.py),
    which requires real transition dipole vectors for its isotropic
    orientational-averaging formulas. Flagged explicitly as an addition on
    top of the cited literature method, using the same physics already
    validated in amide2dir_constants.py.
    """
    Q01 = np.sqrt(16.8576 / omega_cm1)
    return mu_derivative_vector * Q01
