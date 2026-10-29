# File guide

| Manuscript item | Files |
|---|---|
| Reference model, LCH-Seg and binary union, fivefold (Fig. 1b,c) | `figure_source_data/figure_1_source_data.csv`; `locked_binary_or` rows of `results/matched_fp/matched_oof_threshold_sweep_summary.csv` |
| Candidate burden per patient | `results/candidate_burden/` |
| Matched false-positive comparison | `results/matched_fp/` |
| Detection by component size and overlap threshold (Fig. 2d) | `results/robustness/tiny_endpoint_robustness_*.csv` |
| Voxel volume of the tiny-lesion endpoint | `results/robustness/voxel_volume_*.csv` |
| Supportive full-volume configuration (Fig. 2b,c) | `results/ablation/final_full_volume_*.csv` |
| External zero-shot testing (Fig. 4c,d) | `results/external_zero_shot/` |
| Blinded reader review | `results/reader_review/` |
| Figure source data | `figure_source_data/figure_1_source_data.csv` to `figure_5_source_data.csv` |

The fivefold comparison uses 80 patients and 681 ground-truth components of
1-10 voxels. The supportive full-volume configuration uses the same patients
but a separate set of predictions, with 408 such components. It combines
tiny-recall training with expert branches and a histogram gradient-boosting
candidate filter. The two sets are reported separately.

External testing used 60 Kaggle/Mendeley cases and 22 MSLesSeg-2024 cases.
