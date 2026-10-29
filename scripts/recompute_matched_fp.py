from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATIENT = ROOT / "results/matched_fp/matched_oof_threshold_sweep_by_patient.csv"
SUMMARY = ROOT / "results/matched_fp/matched_oof_threshold_sweep_summary.csv"
OUT = ROOT / "outputs/matched_fp_interpolated_comparison_recomputed.csv"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def interp_y(rows, x_key, y_key, target):
    collapsed = defaultdict(list)
    for row in rows:
        collapsed[float(row[x_key])].append(float(row[y_key]))
    xs = np.asarray(sorted(collapsed), dtype=float)
    ys = np.asarray([np.mean(collapsed[x]) for x in xs], dtype=float)
    if target <= xs[0]:
        return float(ys[0])
    if target >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(target, xs, ys))


patient_rows = read(PATIENT)
summary_rows = read(SUMMARY)
ref_rows = [row for row in summary_rows if row["method"] == "reference_threshold"]
lch_rows = [row for row in summary_rows if row["method"] == "lch_threshold"]
union_row = [row for row in summary_rows if row["method"] == "union_output"][0]

cases = sorted({row["case"] for row in patient_rows})
thresholds = sorted({float(row["threshold"]) for row in patient_rows if row["method"] == "reference_threshold"})
lookup = {
    (row["case"], row["method"], float(row["threshold"])): row
    for row in patient_rows
    if row["method"] in {"reference_threshold", "lch_threshold"}
}


def bootstrap(x_key, target, n_boot):
    rng = np.random.default_rng(20260520)
    differences = []
    for _ in range(n_boot):
        sample = rng.choice(cases, size=len(cases), replace=True)
        curves = {}
        for method in ("reference_threshold", "lch_threshold"):
            method_rows = []
            for threshold in thresholds:
                selected = [lookup[(case, method, threshold)] for case in sample]
                x_raw = "fp_components" if x_key == "fp_components_per_patient" else "fp_voxels"
                tiny_hit = sum(int(row["tiny_hit"]) for row in selected)
                tiny_total = sum(int(row["tiny_total"]) for row in selected)
                method_rows.append(
                    {
                        x_key: np.mean([float(row[x_raw]) for row in selected]),
                        "tiny_detection": tiny_hit / tiny_total,
                    }
                )
            curves[method] = method_rows
        differences.append(
            interp_y(curves["lch_threshold"], x_key, "tiny_detection", target)
            - interp_y(curves["reference_threshold"], x_key, "tiny_detection", target)
        )
    values = np.asarray(differences)
    return values.mean(), np.quantile(values, 0.025), np.quantile(values, 0.975)


def locked(rows, threshold, key):
    return float([row for row in rows if abs(float(row["threshold"]) - threshold) < 1e-12][0][key])


targets = [
    ("reference_locked_fp_components", "fp_components_per_patient", locked(ref_rows, 0.5, "fp_components_per_patient_mean")),
    ("lch_locked_fp_components", "fp_components_per_patient", locked(lch_rows, 0.5, "fp_components_per_patient_mean")),
    ("union_fp_components", "fp_components_per_patient", float(union_row["fp_components_per_patient_mean"])),
    ("reference_locked_fp_voxels", "fp_voxels_per_patient", locked(ref_rows, 0.5, "fp_voxels_per_patient_mean")),
    ("lch_locked_fp_voxels", "fp_voxels_per_patient", locked(lch_rows, 0.5, "fp_voxels_per_patient_mean")),
    ("union_fp_voxels", "fp_voxels_per_patient", float(union_row["fp_voxels_per_patient_mean"])),
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-bootstrap",
    type=int,
    default=1000,
    help="Bootstrap iterations. Use 10000 to reproduce the archived manuscript table.",
)
args = parser.parse_args()

out = []
for name, metric, target in targets:
    ref = interp_y(ref_rows, f"{metric}_mean", "tiny_detection_mean", target)
    lch = interp_y(lch_rows, f"{metric}_mean", "tiny_detection_mean", target)
    mean, low, high = bootstrap(metric, target, args.n_bootstrap)
    out.append(
        {
            "target_name": name,
            "burden_metric": metric,
            "target_value": target,
            "reference_tiny_detection_interpolated": ref,
            "lch_tiny_detection_interpolated": lch,
            "lch_minus_reference": lch - ref,
            "bootstrap_mean_difference": mean,
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "n_bootstrap": args.n_bootstrap,
        }
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(out[0]))
    writer.writeheader()
    writer.writerows(out)

for row in out:
    print(row)
