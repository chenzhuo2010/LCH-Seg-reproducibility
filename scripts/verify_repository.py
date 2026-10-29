from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


required = [
    "configs/analysis_protocol.json",
    "data_splits/official_patient_assignments.json",
    "results/candidate_burden/patient_candidate_burden_summary.csv",
    "results/ablation/final_full_volume_by_fold.csv",
    "results/ablation/final_full_volume_by_patient.csv",
    "results/ablation/final_full_volume_summary.csv",
    "results/matched_fp/matched_fp_interpolated_comparison.csv",
    "results/robustness/tiny_endpoint_robustness_summary.csv",
    "results/external_zero_shot/kaggle_mendeley_summary.csv",
    "results/external_zero_shot/mslesseg_summary.csv",
    "results/reader_review/reader_candidate_scores_deidentified.csv",
    "results/reader_review/reader_directional_exact_tests.csv",
]

missing = [rel for rel in required if not (ROOT / rel).is_file()]
assert not missing, f"missing files: {missing}"

protocol = json.loads((ROOT / "configs/analysis_protocol.json").read_text(encoding="utf-8"))
assert protocol["primary_internal_analysis"]["patients"] == 80
assert protocol["primary_internal_analysis"]["ground_truth_tiny_components"] == 681
objective = protocol["component_hit_objective"]
assert objective["checkpoint_trainer"] == "nnUNetTrainerTinyComponentClean1000epochs"
assert objective["base_loss"] == "nnU-Net Dice plus cross entropy"
assert abs(float(objective["component_weight"]) - 0.25) < 1e-12
assert objective["training_component_definition"]["size_range_pixels"] == [1, 10]
assert objective["evaluation_component_definition"]["connectivity"] == 26

full_volume_fold = rows("results/ablation/final_full_volume_by_fold.csv")
full_volume_patient = rows("results/ablation/final_full_volume_by_patient.csv")
full_volume_summary = rows("results/ablation/final_full_volume_summary.csv")
assert len(full_volume_fold) == 5
assert len(full_volume_patient) == 80
assert len(full_volume_summary) == 1
assert {row["method"] for row in full_volume_patient} == {"final_full_volume"}

candidate = {
    row["method"]: row
    for row in rows("results/candidate_burden/patient_candidate_burden_summary.csv")
}
assert abs(float(candidate["lch_binary_union"]["additional_pred_components_vs_baseline_mean"]) - 6.1375) < 1e-12
assert abs(float(candidate["baseline_oof"]["fp_components_mean"]) - 6.7375) < 1e-12
assert abs(float(candidate["lch_binary_union"]["fp_components_mean"]) - 11.95) < 1e-12

matched = {
    row["target_name"]: row
    for row in rows("results/matched_fp/matched_fp_interpolated_comparison.csv")
}
voxel = matched["reference_locked_fp_voxels"]
assert abs(float(voxel["lch_minus_reference"]) - 0.1064082866136461) < 1e-12
assert abs(float(voxel["bootstrap_ci95_low"]) - 0.07891705245242155) < 1e-12
assert abs(float(voxel["bootstrap_ci95_high"]) - 0.15088371860185612) < 1e-12

robustness = rows("results/robustness/tiny_endpoint_robustness_summary.csv")
lookup = {
    (row["method"], row["size_bin"], row["overlap_definition"]): row
    for row in robustness
}
assert lookup[("reference_model", "3-10_voxels", "any_overlap")]["detected_components"] == "436"
assert lookup[("lch_seg", "3-10_voxels", "any_overlap")]["detected_components"] == "510"
assert lookup[("reference_model", "3-10_voxels", "ge20pct_overlap")]["detected_components"] == "433"
assert lookup[("lch_seg", "3-10_voxels", "ge20pct_overlap")]["detected_components"] == "497"

reader = rows("results/reader_review/reader_candidate_scores_deidentified.csv")
def yes(category: str, key: str) -> int:
    return sum(row[key].lower() == "true" for row in reader if row["category"] == category)

assert yes("recovered_tiny", "review_yes") == 9
assert yes("new_fp", "review_yes") == 4
assert yes("recovered_tiny", "true_ms_yes") == 8
assert yes("new_fp", "true_ms_yes") == 3

directional = rows("results/reader_review/reader_directional_exact_tests.csv")
assert {row["comparison"] for row in directional} == {"recovered_tiny vs new_fp"}
assert "two_sided_fisher_p" not in directional[0], "only one-sided tests are reported"
reported_bins = {"1-10_voxels", "1-2_voxels", "3-5_voxels", "6-10_voxels", "3-10_voxels"}
assert {row["size_bin"] for row in robustness} == reported_bins

expected_result_dirs = {
    "ablation",
    "candidate_burden",
    "external_zero_shot",
    "matched_fp",
    "reader_review",
    "robustness",
}
actual_result_dirs = {path.name for path in (ROOT / "results").iterdir() if path.is_dir()}
assert actual_result_dirs == expected_result_dirs, (
    f"unexpected result directories: {sorted(actual_result_dirs - expected_result_dirs)}"
)

expected_scripts = {
    "nnunet_trainer_component_hit.py",
    "recompute_matched_fp.py",
    "recompute_reader_tests.py",
    "verify_repository.py",
}
actual_scripts = {path.name for path in (ROOT / "scripts").glob("*.py")}
assert actual_scripts == expected_scripts, f"unexpected scripts: {sorted(actual_scripts - expected_scripts)}"

expected_figure_sources = {f"figure_{index}_source_data.csv" for index in range(1, 6)}
actual_figure_sources = {path.name for path in (ROOT / "figure_source_data").iterdir() if path.is_file()}
assert actual_figure_sources == expected_figure_sources, (
    f"unexpected figure-source files: {sorted(actual_figure_sources - expected_figure_sources)}"
)

for path in ROOT.rglob("*"):
    if ".git" in path.parts or "outputs" in path.parts:
        continue
    assert "__pycache__" not in path.parts, f"Python cache directory in {path.relative_to(ROOT)}"
    assert path.suffix.lower() not in {".pyc", ".pyo"}, f"compiled Python artifact in {path.relative_to(ROOT)}"

manifest_rows = rows("MANIFEST.csv")
manifest = {row["path"]: row for row in manifest_rows}
actual_files = {
    path.relative_to(ROOT).as_posix(): path
    for path in ROOT.rglob("*")
    if path.is_file()
    and ".git" not in path.parts
    and "outputs" not in path.parts
    and path.name != "MANIFEST.csv"
}
assert set(manifest) == set(actual_files), "MANIFEST.csv does not match repository files"
for relative, path in actual_files.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert int(manifest[relative]["size_bytes"]) == path.stat().st_size
    assert manifest[relative]["sha256"] == digest

print("PASS: repository structure, reported values, and file checksums")
