"""
compute_map_coupling.py
========================
Alternative to extract_amide_modes.py + calculate_geometry.py +
calculate_coupling.py: computes omega_A, omega_B, and J_AB DIRECTLY from
raw MD .gro snapshots + a GROMACS topology (partial charges), using the
Wang, Middleton, Zanni & Skinner (2011, JPCB 115, 3713) electrostatic
frequency map and TDC coupling scheme -- no Gaussian QM frequency
calculation needed at all.

Output is schema-compatible with calculate_hamiltonian.py's expected
input, so the rest of the pipeline (Hamiltonian diagonalization,
transition dipoles, 2D-IR response function) is reused UNCHANGED.

Usage (single snapshot, for testing):
    python compute_map_coupling.py --gro npt.gro --top topol.top \
        --resA 25 --resB 73 --out amide_coupling_data.csv

Usage (directory of per-snapshot .gro files, matching snapshot numbering):
    python compute_map_coupling.py --grodir /path/to/snapshots --pattern "*.gro" \
        --top topol.top --resA 25 --resB 73 --out amide_coupling_data.csv
"""

import argparse
import glob
import re
import sys

import numpy as np
import pandas as pd

from gro_reader import read_gro
from topology_charges import build_full_charge_array, parse_bonds, build_exclusion_sets
from map_physics import (
    compute_backbone_frequency, transition_dipole_vector, tdc_coupling,
    transition_dipole_debye,
)


def natural_key(filename):
    numbers = re.findall(r"\d+", filename.split("/")[-1])
    return int(numbers[-1]) if numbers else 0


def get_backbone_atoms(resnums, resnames, atomnames, resnum):
    """Return dict {C,O,N,H: 0-based index} for the backbone amide unit of
    the given residue number. Proline has no amide H (N bonded to 2 C's)."""
    idx = {}
    mask = resnums == resnum
    for name in ["C", "O", "N", "H"]:
        candidates = np.where(mask & (atomnames == name))[0]
        if len(candidates) > 0:
            idx[name] = candidates[0]
    return idx


def get_residue_atom_mask(resnums, resnum):
    """Boolean mask (len = n_atoms) for ALL atoms belonging to a given
    residue number. Used to additionally exclude a chromophore's immediate
    sequence neighbors (i-1, i+1) from the raw electrostatic field sum --
    see --exclude-neighbors flag and README for why this matters."""
    return resnums == resnum


def process_coords(coords, box, resnums, resnames, atomnames, charges,
                    exclusion_sets, n_protein, resA, resB, cutoff,
                    exclude_neighbors=True):
    """Core physics, independent of where coords/box came from (.gro file
    or a trajectory frame)."""
    atoms_A = get_backbone_atoms(resnums, resnames, atomnames, resA)
    atoms_B = get_backbone_atoms(resnums, resnames, atomnames, resB)

    required = ["C", "O", "N"]
    for label, atoms in [("A", atoms_A), ("B", atoms_B)]:
        missing = [r for r in required if r not in atoms]
        if missing:
            raise ValueError(f"Residue {label}: missing backbone atom(s) {missing}")

    C_A, O_A, N_A = coords[atoms_A["C"]], coords[atoms_A["O"]], coords[atoms_A["N"]]
    C_B, O_B, N_B = coords[atoms_B["C"]], coords[atoms_B["O"]], coords[atoms_B["N"]]
    H_A = coords[atoms_A["H"]] if "H" in atoms_A else None  # Proline: no H
    H_B = coords[atoms_B["H"]] if "H" in atoms_B else None

    def exclusion_mask(atom_0based, own_resnum):
        atom_1based = atom_0based + 1  # topology bonds are 1-based
        mask = np.zeros(len(coords), dtype=bool)
        if atom_1based in exclusion_sets:
            excl_1based = np.array(list(exclusion_sets[atom_1based]))
            excl_0based = excl_1based[excl_1based <= n_protein] - 1
            mask[excl_0based] = True
        if exclude_neighbors:
            mask |= get_residue_atom_mask(resnums, own_resnum - 1)
            mask |= get_residue_atom_mask(resnums, own_resnum + 1)
        return mask

    excl_C_A = exclusion_mask(atoms_A["C"], resA)
    excl_N_A = exclusion_mask(atoms_A["N"], resA)
    excl_C_B = exclusion_mask(atoms_B["C"], resB)
    excl_N_B = exclusion_mask(atoms_B["N"], resB)

    omega_A, EC_A, EN_A = compute_backbone_frequency(
        C_A, O_A, N_A, H_A, coords, charges, box, excl_C_A, excl_N_A, cutoff)
    omega_B, EC_B, EN_B = compute_backbone_frequency(
        C_B, O_B, N_B, H_B, coords, charges, box, excl_C_B, excl_N_B, cutoff)

    origin_A, mu_A_deriv = transition_dipole_vector(C_A, O_A, N_A)
    origin_B, mu_B_deriv = transition_dipole_vector(C_B, O_B, N_B)

#    J_AB = tdc_coupling(origin_A, mu_A_deriv, origin_B, mu_B_deriv, box)
    J_AB, kappa = tdc_coupling(origin_A, mu_A_deriv, origin_B, mu_B_deriv, box)

    mu_A_debye_vec = transition_dipole_debye(mu_A_deriv, omega_A)
    mu_B_debye_vec = transition_dipole_debye(mu_B_deriv, omega_B)
    mu_A_debye = np.linalg.norm(mu_A_debye_vec)
    mu_B_debye = np.linalg.norm(mu_B_debye_vec)
    muA_hat = mu_A_debye_vec / mu_A_debye
    muB_hat = mu_B_debye_vec / mu_B_debye

    from map_physics import minimum_image
    R_AB_vec = minimum_image(origin_B - origin_A, box)
    R_AB = np.linalg.norm(R_AB_vec)

    return {
        "omega_A": omega_A, "omega_B": omega_B, "J_AB_cm1": J_AB,
        "EC_A": EC_A, "EN_A": EN_A, "EC_B": EC_B, "EN_B": EN_B,
        "mu_A_Debye": mu_A_debye, "mu_B_Debye": mu_B_debye,
        "muA_hat_x": muA_hat[0], "muA_hat_y": muA_hat[1], "muA_hat_z": muA_hat[2],
        "muB_hat_x": muB_hat[0], "muB_hat_y": muB_hat[1], "muB_hat_z": muB_hat[2],
        "R_AB_Angstrom": R_AB,
        "kappa": kappa,
    }


def process_snapshot(gro_path, charges, exclusion_sets, n_protein, resA, resB, cutoff,
                      exclude_neighbors=True):
    """.gro-file version: reads the file, then calls process_coords."""
    coords, box, resnums, resnames, atomnames = read_gro(gro_path)
    return process_coords(coords, box, resnums, resnames, atomnames, charges,
                           exclusion_sets, n_protein, resA, resB, cutoff,
                           exclude_neighbors)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gro", default=None, help="Single .gro snapshot (for testing)")
    p.add_argument("--grodir", default=None, help="Directory of per-snapshot .gro files")
    p.add_argument("--pattern", default="*.gro")
    p.add_argument("--traj", default=None,
                    help="Trajectory file (.xtc or .trr) to read frames directly from "
                         "(requires MDAnalysis: pip install MDAnalysis). Use with --struct.")
    p.add_argument("--struct", default=None,
                    help="Structure/topology file matching --traj's atom order "
                         "(e.g. the same .gro you've used elsewhere, or a .tpr). "
                         "Required if --traj is given.")
    p.add_argument("--start", type=int, default=0, help="First frame index (0-based) for --traj")
    p.add_argument("--stop", type=int, default=None, help="Stop before this frame index for --traj")
    p.add_argument("--step", type=int, default=1, help="Frame stride for --traj")
    p.add_argument("--start-time-ps", type=float, default=None,
                    help="Alternative to --start/--stop: select frames by simulation time (ps)")
    p.add_argument("--stop-time-ps", type=float, default=None)
    p.add_argument("--min-distance", type=float, default=None,
                    help="Only process frames where the backbone C=O midpoint "
                         "distance (PBC-corrected) between residues A and B is "
                         "AT LEAST this value (Angstrom). Combine with "
                         "--max-distance to select a clean structural population "
                         "directly (e.g. --min-distance 60 for a 'far apart' "
                         "ensemble) rather than guessing a time window -- this "
                         "also automatically excludes single-frame PBC/trajectory "
                         "artifacts (e.g. a spurious multi-thousand-Angstrom jump), "
                         "since those won't fall in your chosen real-distance range.")
    p.add_argument("--max-distance", type=float, default=None,
                    help="Only process frames where the distance is AT MOST this "
                         "value (Angstrom). See --min-distance.")
    p.add_argument("--top", required=True, help="GROMACS topol.top")
    p.add_argument("--resA", type=int, required=True, help="Residue number of chromophore A")
    p.add_argument("--resB", type=int, required=True, help="Residue number of chromophore B")
    p.add_argument("--cutoff", type=float, default=20.0, help="Electric field cutoff (Angstrom)")
    p.add_argument("--no-exclude-neighbors", action="store_true",
                    help="Do NOT exclude i-1/i+1 sequence-neighbor residues "
                         "from the electrostatic field sum (not recommended "
                         "-- see README; default is to exclude them since "
                         "including them via raw Coulomb is known-wrong, "
                         "though excluding them without the proper NNFS "
                         "replacement is also an approximation).")
    p.add_argument("--out", default="amide_coupling_data.csv")
    args = p.parse_args()

    print("Parsing bonded topology for nonbonded-exclusion list (nrexcl=3)...")
    from topology_charges import parse_protein_charges
    n_protein, _ = parse_protein_charges(args.top)
    bonds = parse_bonds(args.top)
    exclusion_sets = build_exclusion_sets(bonds, n_protein, nrexcl=3)
    print(f"Exclusion graph built for {n_protein} protein atoms, {len(bonds)} bonds.")

    results = []

    if args.traj:
        if not args.struct:
            print("--struct is required when using --traj")
            sys.exit(1)
        from trajectory_reader import iterate_trajectory

        print(f"Reading atom identity from --struct: {args.struct}")
        _, _, resnums, resnames, atomnames = read_gro(args.struct)
        print("Building charge array from topology...")
        charges = build_full_charge_array(args.top, len(resnums))
        print(f"Charge array built: {len(charges)} atoms, net charge = {charges.sum():.4f}")

        print(f"Iterating trajectory: {args.traj}")
        for frame_index, time_ps, coords, box in iterate_trajectory(
                args.struct, args.traj, args.start, args.stop, args.step,
                args.start_time_ps, args.stop_time_ps):

            if args.min_distance is not None or args.max_distance is not None:
                # Cheap pre-filter: compute just the backbone C=O midpoint
                # distance (PBC-corrected) BEFORE doing the expensive field
                # calculation, and skip this frame entirely if it's outside
                # the requested range. This also naturally excludes single-
                # frame trajectory/PBC artifacts (e.g. a spurious multi-
                # thousand-Angstrom jump), since those won't fall in any real
                # chosen distance range.
                atoms_A_quick = get_backbone_atoms(resnums, resnames, atomnames, args.resA)
                atoms_B_quick = get_backbone_atoms(resnums, resnames, atomnames, args.resB)
                mid_A = 0.5 * (coords[atoms_A_quick["C"]] + coords[atoms_A_quick["O"]])
                mid_B = 0.5 * (coords[atoms_B_quick["C"]] + coords[atoms_B_quick["O"]])
                from map_physics import minimum_image
                quick_dist = np.linalg.norm(minimum_image(mid_B - mid_A, box))
                if args.min_distance is not None and quick_dist < args.min_distance:
                    continue
                if args.max_distance is not None and quick_dist > args.max_distance:
                    continue

            print(f"Processing frame {frame_index} (t={time_ps:.2f} ps)")
            try:
                result = process_coords(coords, box, resnums, resnames, atomnames,
                                         charges, exclusion_sets, n_protein,
                                         args.resA, args.resB, args.cutoff,
                                         exclude_neighbors=not args.no_exclude_neighbors)
                result["snapshot"] = frame_index
                result["file"] = f"{args.traj}@t={time_ps:.2f}ps"
                results.append(result)
                print(f"  omega_A={result['omega_A']:.2f}  omega_B={result['omega_B']:.2f}  "
                      f"J_AB={result['J_AB_cm1']:.4f} cm^-1  R_AB={result['R_AB_Angstrom']:.2f} A")
            except Exception as e:
                print(f"  ERROR: {e}")

    else:
        if args.gro:
            files = [args.gro]
        elif args.grodir:
            files = sorted(glob.glob(f"{args.grodir}/{args.pattern}"), key=natural_key)
        else:
            print("Must specify --gro, --grodir, or --traj (with --struct)")
            sys.exit(1)

        print(f"Found {len(files)} snapshot(s)")

        coords0, _, _, _, _ = read_gro(files[0])
        print("Building charge array from topology...")
        charges = build_full_charge_array(args.top, len(coords0))
        print(f"Charge array built: {len(charges)} atoms, net charge = {charges.sum():.4f}")

        for snapshot, path in enumerate(files):
            print(f"Processing {snapshot + 1}/{len(files)}: {path}")
            try:
                result = process_snapshot(path, charges, exclusion_sets, n_protein,
                                           args.resA, args.resB, args.cutoff,
                                           exclude_neighbors=not args.no_exclude_neighbors)
                result["snapshot"] = snapshot
                result["file"] = path
                results.append(result)
                print(f"  omega_A={result['omega_A']:.2f}  omega_B={result['omega_B']:.2f}  "
                      f"J_AB={result['J_AB_cm1']:.4f} cm^-1  R_AB={result['R_AB_Angstrom']:.2f} A")
            except Exception as e:
                print(f"  ERROR: {e}")

    if not results:
        print("No snapshots succeeded.")
        sys.exit(1)

    df = pd.DataFrame(results)
    cols = ["snapshot", "file", "omega_A", "omega_B", "J_AB_cm1",
            "mu_A_Debye", "mu_B_Debye",
            "muA_hat_x", "muA_hat_y", "muA_hat_z",
            "muB_hat_x", "muB_hat_y", "muB_hat_z",
            "R_AB_Angstrom", "kappa", "EC_A", "EN_A", "EC_B", "EN_B"]
    df = df[cols]
    df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}")
    print(df[["omega_A", "omega_B", "J_AB_cm1", "R_AB_Angstrom"]].describe())


if __name__ == "__main__":
    main()
