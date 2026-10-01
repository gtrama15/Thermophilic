"""
plot_distance_vs_time.py
==========================
Fast, lightweight diagnostic: plots the actual R_AB(t) trajectory for a
residue pair (backbone C=O midpoint distance, PBC-corrected), WITHOUT
computing the full electrostatic map/coupling for every frame. Use this
to see directly how the distance evolves -- transient re-approaches,
gradual separation, sudden jumps -- before choosing any time-window split
for ensemble comparison.

Usage:
    python plot_distance_vs_time.py --traj pull_80.xtc --struct 80.gro \
        --resA 81 --resB 100 --out distance_vs_time.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

from map_physics import minimum_image

p = argparse.ArgumentParser()
p.add_argument("--traj", required=True)
p.add_argument("--struct", required=True)
p.add_argument("--resA", type=int, required=True)
p.add_argument("--resB", type=int, required=True)
p.add_argument("--out", default="distance_vs_time.png")
p.add_argument("--csv-out", default="distance_vs_time.csv")
args = p.parse_args()

u = mda.Universe(args.struct, args.traj)

selA_C = u.select_atoms(f"resid {args.resA} and name C")
selA_O = u.select_atoms(f"resid {args.resA} and name O")
selB_C = u.select_atoms(f"resid {args.resB} and name C")
selB_O = u.select_atoms(f"resid {args.resB} and name O")

times, distances = [], []
for ts in u.trajectory:
    box = np.array(ts.dimensions[:3], dtype=float)
    midpoint_A = 0.5 * (selA_C.positions[0] + selA_O.positions[0])
    midpoint_B = 0.5 * (selB_C.positions[0] + selB_O.positions[0])
    d = minimum_image(midpoint_B - midpoint_A, box)
    distances.append(np.linalg.norm(d))
    times.append(ts.time)

times = np.array(times)
distances = np.array(distances)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(times, distances, "-", linewidth=1)
ax.set_xlabel("Time (ps)")
ax.set_ylabel(f"R_AB, residues {args.resA}-{args.resB} (Angstrom)")
ax.set_title(f"Backbone C=O midpoint distance vs. time")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"Saved: {args.out}")

import pandas as pd
pd.DataFrame({"time_ps": times, "R_AB_Angstrom": distances}).to_csv(args.csv_out, index=False)
print(f"Saved: {args.csv_out}")
print(f"\nR_AB range: {distances.min():.2f} - {distances.max():.2f} Angstrom")
print(f"R_AB at t=0: {distances[0]:.2f}, at t={times[-1]:.0f}ps: {distances[-1]:.2f}")
