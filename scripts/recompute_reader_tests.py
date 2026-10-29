from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/reader_review/reader_candidate_scores_deidentified.csv"
OUT = ROOT / "outputs/reader_directional_exact_tests_recomputed.csv"

with SRC.open(newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))

tests = [
    ("review_yes", "greater"),
    ("true_ms_yes", "greater"),
    ("fp_or_artifact_yes", "less"),
]
out = []
for endpoint, alternative in tests:
    recovered = [row[endpoint].lower() == "true" for row in rows if row["category"] == "recovered_tiny"]
    no_overlap = [row[endpoint].lower() == "true" for row in rows if row["category"] == "new_fp"]
    table = [
        [sum(recovered), len(recovered) - sum(recovered)],
        [sum(no_overlap), len(no_overlap) - sum(no_overlap)],
    ]
    odds, p = fisher_exact(table, alternative=alternative)
    out.append(
        {
            "endpoint": endpoint,
            "comparison": "recovered_tiny_vs_no_overlap_added",
            "alternative": alternative,
            "table": str(table),
            "odds_ratio": odds,
            "one_sided_fisher_p": p,
        }
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(out[0]))
    writer.writeheader()
    writer.writerows(out)

for row in out:
    print(row)
