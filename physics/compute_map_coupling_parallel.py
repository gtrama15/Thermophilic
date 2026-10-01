"""
compute_map_coupling_parallel.py
==================================
Parallelized version of compute_map_coupling.py's trajectory mode. Each
frame's calculation is fully independent (same static charges/exclusions,
different coordinates), so this splits the trajectory into contiguous
chunks and processes them across multiple CPU cores.

Each worker process opens its OWN MDAnalysis Universe handle (Universe
objects aren't easily shared across processes) and iterates its assigned
contiguous frame range sequentially -- more efficient than random-access
seeking, and avoids passing large per-frame coordinate arrays between
processes.

IMPORTANT: this was developed and syntax/logic-reviewed in a single-core
sandbox (could not demonstrate real multi-core speedup there). Please
confirm on your own multi-core machine: run with --workers 1 and --workers
N on a SMALL frame range first and confirm identical output, before
trusting it for a full production run.

Usage:
    python compute_map_coupling_parallel.py --traj pull_80.xtc --struct 80.gro \
        --top topol.top --resA 2 --resB 27 --min-distance 80 \
        --workers 8 --out ensemble_2_27_far.csv
"""

import argparse
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from map_physics import minimum_image
from compute_map_coupling import get_backbone_atoms, process_coords
from topology_charges import build_full_charge_array, parse_bonds, build_exclusion_sets, parse_protein_charges

# Per-worker global state, set once by _worker_init, reused across all
# frames that worker processes (avoids re-pickling large arrays per task).
_worker_state = {}


def _worker_init(struct, traj, charges, exclusion_sets, n_protein, resA, resB,
                  cutoff, exclude_neighbors, min_distance, max_distance):
    import MDAnalysis as mda
    _worker_state["u"] = mda.Universe(struct, traj)
    _worker_state["charges"] = charges
    _worker_state["exclusion_sets"] = exclusion_sets
    _worker_state["n_protein"] = n_protein
    _worker_state["resA"] = resA
    _worker_state["resB"] = resB
    _worker_state["cutoff"] = cutoff
    _worker_state["exclude_neighbors"] = exclude_neighbors
    _worker_state["min_distance"] = min_distance
    _worker_state["max_distance"] = max_distance
    # atom identity (resnums/resnames/atomnames) is static -- read once
    from gro_reader import read_gro
    _, _, resnums, resnames, atomnames = read_gro(struct)
    _worker_state["resnums"] = resnums
    _worker_state["resnames"] = resnames
    _worker_state["atomnames"] = atomnames


def _process_chunk(frame_range):
    """Process a contiguous range of frame indices [start, stop). Returns
    a list of result dicts (only for frames that passed the distance filter
    and processed without error)."""
    start, stop = frame_range
    u = _worker_state["u"]
    resnums = _worker_state["resnums"]
    resnames = _worker_state["resnames"]
    atomnames = _worker_state["atomnames"]
    resA, resB = _worker_state["resA"], _worker_state["resB"]
    min_d, max_d = _worker_state["min_distance"], _worker_state["max_distance"]

    results = []
    for ts in u.trajectory[start:stop]:
        frame_index = ts.frame
        time_ps = ts.time
        coords = ts.positions.copy()
        box = np.array(ts.dimensions[:3], dtype=float)

        if min_d is not None or max_d is not None:
            atoms_A_q = get_backbone_atoms(resnums, resnames, atomnames, resA)
            atoms_B_q = get_backbone_atoms(resnums, resnames, atomnames, resB)
            mid_A = 0.5 * (coords[atoms_A_q["C"]] + coords[atoms_A_q["O"]])
            mid_B = 0.5 * (coords[atoms_B_q["C"]] + coords[atoms_B_q["O"]])
            quick_dist = np.linalg.norm(minimum_image(mid_B - mid_A, box))
            if min_d is not None and quick_dist < min_d:
                continue
            if max_d is not None and quick_dist > max_d:
                continue

        try:
            result = process_coords(
                coords, box, resnums, resnames, atomnames,
                _worker_state["charges"], _worker_state["exclusion_sets"],
                _worker_state["n_protein"], resA, resB, _worker_state["cutoff"],
                exclude_neighbors=_worker_state["exclude_neighbors"],
            )
            result["snapshot"] = frame_index
            result["file"] = f"frame{frame_index}@t={time_ps:.2f}ps"
            results.append(result)
        except Exception as e:
            print(f"  [worker] frame {frame_index}: ERROR: {e}", file=sys.stderr)

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", required=True)
    p.add_argument("--struct", required=True)
    p.add_argument("--top", required=True)
    p.add_argument("--resA", type=int, required=True)
    p.add_argument("--resB", type=int, required=True)
    p.add_argument("--cutoff", type=float, default=20.0)
    p.add_argument("--no-exclude-neighbors", action="store_true")
    p.add_argument("--min-distance", type=float, default=None)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--start-time-ps", type=float, default=None)
    p.add_argument("--stop-time-ps", type=float, default=None)
    p.add_argument("--workers", type=int, default=mp.cpu_count(),
                    help="Number of parallel worker processes (default: all available cores)")
    p.add_argument("--out", default="amide_coupling_data.csv")
    args = p.parse_args()

    import MDAnalysis as mda
    u_main = mda.Universe(args.struct, args.traj)
    n_frames = len(u_main.trajectory)

    if args.start_time_ps is not None or args.stop_time_ps is not None:
        print("Scanning frame times (metadata only, fast)...")
        times = np.array([ts.time for ts in u_main.trajectory])
        valid = np.ones(n_frames, dtype=bool)
        if args.start_time_ps is not None:
            valid &= times >= args.start_time_ps
        if args.stop_time_ps is not None:
            valid &= times <= args.stop_time_ps
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            print("No frames in the requested time range.")
            sys.exit(1)
        frame_lo, frame_hi = valid_indices.min(), valid_indices.max() + 1
    else:
        frame_lo, frame_hi = 0, n_frames

    print(f"Processing frames [{frame_lo}, {frame_hi}) of {n_frames} total, "
          f"using {args.workers} worker(s)")

    print("Building charge array from topology...")
    coords0 = u_main.atoms.positions
    charges = build_full_charge_array(args.top, len(coords0))
    n_protein, _ = parse_protein_charges(args.top)
    bonds = parse_bonds(args.top)
    exclusion_sets = build_exclusion_sets(bonds, n_protein, nrexcl=3)
    print(f"Charge array: {len(charges)} atoms. Exclusion graph: {n_protein} protein atoms, {len(bonds)} bonds.")

    # Split [frame_lo, frame_hi) into --workers contiguous chunks
    n_total = frame_hi - frame_lo
    chunk_size = max(1, n_total // args.workers)
    chunks = []
    pos = frame_lo
    while pos < frame_hi:
        chunks.append((pos, min(pos + chunk_size, frame_hi)))
        pos += chunk_size

    print(f"Split into {len(chunks)} chunk(s): {chunks}")

    init_args = (args.struct, args.traj, charges, exclusion_sets, n_protein,
                 args.resA, args.resB, args.cutoff, not args.no_exclude_neighbors,
                 args.min_distance, args.max_distance)

    if args.workers == 1:
        _worker_init(*init_args)
        all_results = []
        for chunk in chunks:
            all_results.extend(_process_chunk(chunk))
    else:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=init_args) as pool:
            chunk_results = pool.map(_process_chunk, chunks)
        all_results = [r for chunk in chunk_results for r in chunk]

    if not all_results:
        print("No snapshots passed the filter / succeeded.")
        sys.exit(1)

    df = pd.DataFrame(all_results).sort_values("snapshot").reset_index(drop=True)
    cols = ["snapshot", "file", "omega_A", "omega_B", "J_AB_cm1",
            "mu_A_Debye", "mu_B_Debye",
            "muA_hat_x", "muA_hat_y", "muA_hat_z",
            "muB_hat_x", "muB_hat_y", "muB_hat_z",
            "R_AB_Angstrom", "kappa", "EC_A", "EN_A", "EC_B", "EN_B"]
    df = df[cols]
    df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}  ({len(df)} snapshots)")
    print(df[["omega_A", "omega_B", "J_AB_cm1", "R_AB_Angstrom"]].describe())


if __name__ == "__main__":
    main()
