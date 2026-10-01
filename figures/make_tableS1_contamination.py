"""
make_tableS1_contamination.py
===============================
Table S1: contamination-sweep record, saved as a clean CSV -- documents
the data-quality control described in Methods, for every CSV used in the
final analysis.

Usage:
    python make_tableS1_contamination.py --close-config summary_config.json \
        --far-config summary_config_far.json --out final_figures/TableS1_contamination.csv
"""
import argparse
import json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--close-config", required=True)
p.add_argument("--far-config", required=True)
p.add_argument("--extreme-threshold", type=float, default=500.0)
p.add_argument("--out", default="TableS1_contamination.csv")
args = p.parse_args()

rows = []
for state_label, config_path in [("Close", args.close_config), ("Far", args.far_config)]:
    with open(config_path) as f:
        entries = json.load(f)
    for e in entries:
        df = pd.read_csv(e["csv"])
        n = len(df)
        r_min, r_max = df["R_AB_Angstrom"].min(), df["R_AB_Angstrom"].max()
        extreme = df[df["R_AB_Angstrom"] > args.extreme_threshold]
        q1, q3 = df["R_AB_Angstrom"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        mild_tail = df[((df["R_AB_Angstrom"] < lo) | (df["R_AB_Angstrom"] > hi))
                       & (df["R_AB_Angstrom"] <= args.extreme_threshold)]
        status = "CONTAMINATED" if len(extreme) > 0 else ("ok (tail)" if len(mild_tail) > 0 else "OK")
        rows.append({
            "pair": e["pair"], "state": e["state"], "temp": e["temp"],
            "ensemble_state": state_label, "n": n, "status": status,
            "min_R_AB": round(r_min, 2), "max_R_AB": round(r_max, 2),
            "n_extreme_artifacts": len(extreme), "n_mild_tail": len(mild_tail),
        })

table = pd.DataFrame(rows)
table.to_csv(args.out, index=False)
print(table.to_string(index=False))
print(f"\nSaved: {args.out}")
n_bad = (table["status"] == "CONTAMINATED").sum()
if n_bad:
    print(f"\nWARNING: {n_bad} entries still show CONTAMINATED -- clean before finalizing.")
else:
    print("\nAll entries clean.")
