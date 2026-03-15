"""
Aggregate outputs/rownorm_nobias/{dataset}/{run_name}_results.txt
into a single outputs/rownorm_nobias/summary.csv.

Usage:
    python scripts/summarize_nobias.py
"""
import csv
import os
import re

DATASETS  = ["cora", "citeseer", "pubmed", "cornell"]
RUN_NAMES = ["orig", "rownorm_bias", "rownorm_nobias_head", "rownorm_addhiddenbias"]
OUT_ROOT  = "outputs/rownorm_nobias"

# Regex to parse a data row:  method  val  test_full  test_coarse  test_best
_ROW = re.compile(
    r"^(?P<method>\S.*?\S)\s+"
    r"(?P<best_val>\d\.\d+)\s+"
    r"(?P<test_full>\d\.\d+)\s+"
    r"(?P<test_coarse>[\d\.]+|—)\s+"
    r"(?P<test_best>[\d\.]+|—)"
)

rows = []

for ds in DATASETS:
    for run_name in RUN_NAMES:
        path = os.path.join(OUT_ROOT, ds, f"{run_name}_results.txt")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        with open(path) as f:
            for line in f:
                m = _ROW.match(line.strip())
                if not m:
                    continue
                rows.append({
                    "dataset":    ds,
                    "run_name":   run_name,
                    "method":     m.group("method").strip(),
                    "best_val":   m.group("best_val"),
                    "test_full":  m.group("test_full"),
                    "test_coarse":m.group("test_coarse"),
                    "test_best":  m.group("test_best"),
                })

summary_path = os.path.join(OUT_ROOT, "summary.csv")
if rows:
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary_path}  ({len(rows)} rows)")
else:
    print("No result files found — run the experiments first.")
