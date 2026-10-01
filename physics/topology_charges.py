"""
topology_charges.py
====================
Parse a GROMACS topol.top for per-atom partial charges, in .gro atom-index
order, including the protein (explicit [atoms] section) and solvent/ions
(TIP3P water + monatomic ions, standard values -- these are not derived,
they are the universally standard AMBER-compatible TIP3P/ions.itp
parameters).

Assumes (standard GROMACS convention, verified against this specific
topol.top + npt.gro pair): the explicit protein [atoms] block lists atoms
1..N_protein in the same order as they appear in the .gro file, and every
atom after that is water (TIP3P, 3 atoms/molecule: OW, HW1, HW2) followed
by ions (NA+, CL-), in the order given by the [molecules] section at the
end of the topology.
"""

import re
import numpy as np

# Standard AMBER/TIP3P water charges (e), universally standard, not derived
TIP3P_CHARGES = {"OW": -0.834, "HW1": 0.417, "HW2": 0.417}
# Standard monatomic ion charges (e)
ION_CHARGES = {"NA": 1.0, "CL": -1.0}


def parse_protein_charges(top_path):
    """Return (n_protein_atoms, charges_array) for the explicit [atoms] block(s)."""
    charges = []
    in_atoms_section = False
    with open(top_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and "atoms" in stripped.lower():
                in_atoms_section = True
                continue
            if stripped.startswith("[") and "atoms" not in stripped.lower():
                in_atoms_section = False
                continue
            if not in_atoms_section:
                continue
            if not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            try:
                charge = float(parts[6])
            except ValueError:
                continue
            charges.append(charge)
    return len(charges), np.array(charges)


def parse_bonds(top_path):
    """Parse [bonds] section, returns list of (i,j) 1-based atom index pairs."""
    bonds = []
    in_section = False
    with open(top_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and "bonds" in stripped.lower():
                in_section = True
                continue
            if stripped.startswith("["):
                in_section = False
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    bonds.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return bonds


def build_exclusion_sets(bonds, n_atoms, nrexcl=3):
    """
    Build, for every atom (1-based), the set of atoms within `nrexcl-1`
    bonds (i.e. 1-2 and 1-3 neighbors for nrexcl=3, matching this
    topology's declared nrexcl and standard AMBER/GROMACS nonbonded
    exclusion convention) -- these should NEVER be included as external
    "environment" charges when computing the electric field at that atom,
    since real force fields never compute raw nonbonded Coulomb between
    covalently close atoms (that's precisely what nrexcl encodes).

    Returns: dict {1-based atom index: set of 1-based excluded atom indices}
    """
    adjacency = {i: set() for i in range(1, n_atoms + 1)}
    for i, j in bonds:
        if i <= n_atoms and j <= n_atoms:
            adjacency[i].add(j)
            adjacency[j].add(i)

    exclusions = {}
    max_hops = nrexcl - 1  # nrexcl=3 -> exclude up to 2 bonds away (1-2, 1-3)
    for atom in adjacency:
        visited = {atom}
        frontier = {atom}
        for _ in range(max_hops):
            next_frontier = set()
            for a in frontier:
                next_frontier |= adjacency[a]
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
        exclusions[atom] = visited  # includes the atom itself
    return exclusions
    """Return list of (molecule_name, count) from the [molecules] section."""
    molecules = []
    in_section = False
    with open(top_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and "molecules" in stripped.lower():
                in_section = True
                continue
            if stripped.startswith("[") and "molecules" not in stripped.lower():
                in_section = False
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                molecules.append((parts[0], int(parts[1])))
    return molecules


def parse_molecules_section(top_path):
    """Return list of (molecule_name, count) from the [molecules] section."""
    molecules = []
    in_section = False
    with open(top_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and "molecules" in stripped.lower():
                in_section = True
                continue
            if stripped.startswith("[") and "molecules" not in stripped.lower():
                in_section = False
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                molecules.append((parts[0], int(parts[1])))
    return molecules


def build_full_charge_array(top_path, total_atoms):
    """
    Build the full per-atom charge array (length = total_atoms in the .gro
    file), in atom-index order: explicit protein charges from [atoms],
    followed by solvent/ion charges built from the [molecules] system
    composition (TIP3P water = 3 atoms/molecule, monatomic ions = 1
    atom/molecule).
    """
    n_protein, protein_charges = parse_protein_charges(top_path)
    molecules = parse_molecules_section(top_path)

    charge_blocks = [protein_charges]
    running_total = n_protein

    for name, count in molecules:
        name_upper = name.upper()
        if name_upper in ("SOL", "WAT", "TIP3", "TIP3P"):
            block = np.tile(
                [TIP3P_CHARGES["OW"], TIP3P_CHARGES["HW1"], TIP3P_CHARGES["HW2"]],
                count,
            )
            charge_blocks.append(block)
            running_total += count * 3
        elif name_upper in ("NA", "NA+", "SOD"):
            charge_blocks.append(np.full(count, ION_CHARGES["NA"]))
            running_total += count
        elif name_upper in ("CL", "CL-", "CLA"):
            charge_blocks.append(np.full(count, ION_CHARGES["CL"]))
            running_total += count
        elif count == n_protein or "protein" in name_upper.lower():
            # This is the protein moleculetype itself, already counted
            continue
        else:
            raise ValueError(
                f"Unrecognized molecule type '{name}' (count {count}) in "
                f"[molecules] section -- add its charge(s) to this parser."
            )

    full = np.concatenate(charge_blocks)

    if len(full) != total_atoms:
        raise ValueError(
            f"Charge array length ({len(full)}) does not match .gro atom "
            f"count ({total_atoms}). Check the [molecules] section parsing "
            f"and protein atom count ({n_protein})."
        )

    return full
