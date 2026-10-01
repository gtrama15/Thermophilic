"""
gro_reader.py
==============
Fast reader for GROMACS .gro coordinate files. Uses fixed-width column
parsing (the .gro format is strictly fixed-width) via pandas.read_fwf for
speed on large (>1M atom) systems.

.gro format (per atom line):
  columns 0-5:   residue number (int)
  columns 5-10:  residue name
  columns 10-15: atom name
  columns 15-20: atom number (int)
  columns 20-28, 28-36, 36-44: x, y, z (nm)
  (optional velocity columns follow, ignored here)
"""

import numpy as np
import pandas as pd


def read_gro(path):
    """
    Returns:
      coords_angstrom: (N,3) numpy array, coordinates in Angstrom
      box_angstrom: (3,) numpy array, orthorhombic box vectors in Angstrom
      resnums, resnames, atomnames: parallel arrays of length N (metadata)
    """
    with open(path) as f:
        f.readline()  # title
        n_atoms = int(f.readline().strip())

    colspecs = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 28), (28, 36), (36, 44)]
    names = ["resnum", "resname", "atomname", "atomnum", "x", "y", "z"]

    df = pd.read_fwf(
        path, colspecs=colspecs, names=names, skiprows=2, nrows=n_atoms,
        dtype={"resnum": int, "resname": str, "atomname": str, "atomnum": int,
               "x": float, "y": float, "z": float},
    )

    coords_nm = df[["x", "y", "z"]].to_numpy()
    coords_angstrom = coords_nm * 10.0

    with open(path) as f:
        lines = f.readlines()
    box_line = lines[2 + n_atoms].split()
    box_nm = np.array([float(box_line[0]), float(box_line[1]), float(box_line[2])])
    box_angstrom = box_nm * 10.0

    resnames = df["resname"].str.strip().to_numpy()
    atomnames = df["atomname"].str.strip().to_numpy()
    resnums = df["resnum"].to_numpy()

    return coords_angstrom, box_angstrom, resnums, resnames, atomnames
