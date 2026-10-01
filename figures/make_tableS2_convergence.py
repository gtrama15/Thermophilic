"""
make_tableS2_convergence.py
=============================
Table S2: first-half vs. second-half J_AB mean comparison for every
Close and Far ensemble, saved as a clean CSV -- documents the
convergence checks referenced in Methods/Results.

Usage:
    python make_tableS2_convergence.py --close-config summary_config.json \
        --far-config summary_config_far.json --out final_figures/TableS2_convergence.csv
"""
import argparse
import json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--close-config", required=True)
p.add_argument("--far-config", required=True)
p.add_argument("--out", default="TableS2_convergence.csv")
args = p.parse_args()

rows = []
for state_label, config_path in [("Close", args.close_config), ("Far", args.far_config)]:
    with open(config_path) as f:
        entries = json.load(f)
    for e in entries:
        df = pd.read_csv(e["csv"])
        n = len(df)
        half = n // 2
        first, second = df.iloc[:half], df.iloc[half:]
        first_mean = first["J_AB_cm1"].mean()
        second_mean = second["J_AB_cm1"].mean()
        rows.append({
            "pair": e["pair"], "state": e["state"], "temp": e["temp"],
            "ensemble_state": state_label, "n": n,
            "J_AB_first_half": round(first_mean, 4),
            "J_AB_second_half": round(second_mean, 4),
            "abs_diff": round(abs(first_mean - second_mean), 4),
        })

table = pd.DataFrame(rows)
table.to_csv(args.out, index=False)
print(table.to_string(index=False))
print(f"\nSaved: {args.out}")
