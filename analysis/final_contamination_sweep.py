"""
final_contamination_sweep.py
==============================
Final pre-publication check: scans every coupling-data CSV used in your
final figures for the known pre-filter/final-distance-mismatch artifact
(the recurring bug throughout this analysis), using the same physically-
impossible-vs-mild-tail distinction validated in compare_temperatures.py.

Give it a JSON list of {"label": ..., "csv": ...} entries (reuse your
summary_config.json entries directly -- they already have "pair"/"state"/
"temp"/"csv" fields, which this script also understands -- or list every
Close/Far CSV you're using in final figures/tables under a "label" key).

Usage:
    python final_contamination_sweep.py --config all_final_csvs.json
"""
import argparse
import json
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--extreme-threshold", type=float, default=500.0,
                help="R_AB (Angstrom) above which a value is treated as a "
                     "physically impossible artifact for an intra-protein "
                     "pair, not normal variation.")
args = p.parse_args()

with open(args.config) as f:
    entries = json.load(f)

print(f"Sweeping {len(entries)} file(s)...\n")
print(f"{'label':45s} {'n':>6s} {'status':>12s}  {'min R_AB':>10s} {'max R_AB':>10s}  notes")
print("-" * 115)

any_problem = False
for e in entries:
    label = e.get("label") or f"{e.get('pair','')} {e.get('state','')} {e.get('temp','')}".strip()
    try:
        df = pd.read_csv(e["csv"])
    except FileNotFoundError:
        print(f"{label:45s} {'--':>6s} {'MISSING FILE':>12s}  path: {e['csv']}")
        any_problem = True
        continue

    n = len(df)
    r_min, r_max = df["R_AB_Angstrom"].min(), df["R_AB_Angstrom"].max()

    extreme = df[df["R_AB_Angstrom"] > args.extreme_threshold]
    q1, q3 = df["R_AB_Angstrom"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    mild_tail = df[((df["R_AB_Angstrom"] < lo) | (df["R_AB_Angstrom"] > hi))
                   & (df["R_AB_Angstrom"] <= args.extreme_threshold)]

    if len(extreme) > 0:
        status = "CONTAMINATED"
        note = f"{len(extreme)} frame(s) > {args.extreme_threshold} Angstrom -- CLEAN BEFORE USING"
        any_problem = True
    elif len(mild_tail) > 0:
        status = "ok (tail)"
        note = f"{len(mild_tail)} mild statistical-tail value(s), not artifacts"
    else:
        status = "OK"
        note = ""

    print(f"{label:45s} {n:6d} {status:>12s}  {r_min:10.2f} {r_max:10.2f}  {note}")

print()
if any_problem:
    print("*** ACTION NEEDED: one or more files above need attention. ***")
    print("For CONTAMINATED files, clean before finalizing figures, e.g.:")
    print("  df = df[df['R_AB_Angstrom'] < <sane_upper_bound>]  # for a 'Close' ensemble")
    print("  df = df[df['R_AB_Angstrom'] > <sane_lower_bound>]  # for a 'Far' ensemble")
    print("For MISSING FILE entries, fix the path in your config before re-running.")
else:
    print("All files clean. No physically-impossible R_AB values found.")
