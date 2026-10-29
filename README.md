# LCH-Seg reproducibility repository

This repository contains the analysis assets for the manuscript
**Component-hit training improves the small-signal sensitivity of AI-assisted MRI
lesion sensing in multiple sclerosis**.

## Contents

- de-identified patient and fold assignments;
- matched-FP, candidate-burden, tiny-endpoint robustness, supportive full-volume,
  external zero-shot, and reader-review result tables for the analyses reported
  in the manuscript;
- source data for the five manuscript figures;
- structured analysis configurations and environment information;
- the primary component-hit loss and trainer implementation;
- scripts that recompute selected statistics and verify reported values.

Raw MRI volumes, lesion masks, model checkpoints and prediction caches are not
included. Reader-review results are provided as de-identified scores.

## Quick verification

From the repository root:

```bash
python scripts/verify_repository.py
python scripts/recompute_reader_tests.py
python scripts/recompute_matched_fp.py
```

The matched-FP command uses 1,000 bootstrap iterations for a quick public
recomputation. Use `python scripts/recompute_matched_fp.py --n-bootstrap 10000`
to reproduce the reported analysis.

The recomputation scripts write outputs under `outputs/`, which is excluded
from version control.

## Primary component-hit trainer

`scripts/nnunet_trainer_component_hit.py` contains the primary loss definition
used by the matched OOF LCH-Seg checkpoint:

```
DiceCE + 0.25 * component_hit_loss
```

The component term identifies 8-connected foreground components containing
1-10 pixels within each processed axial training patch and rewards at least one
positive prediction within each component. The source can be placed in an
importable nnU-Net trainer directory and invoked with the trainer class
`nnUNetTrainerTinyComponentClean1000epochs`.

## Data access

The imaging datasets are not redistributed. Obtain them from their original
public releases and comply with the source licenses:

- MS3SEG: Figshare article DOI `10.6084/m9.figshare.30393475`
- Kaggle/Mendeley MS lesion segmentation cohort
- MSLesSeg-2024 public release

## Files

See `docs/FILE_GUIDE.md` for which files correspond to each figure and result.
The fivefold comparison and the supportive full-volume configuration use the
same 80 patients but separate predictions, so their results are reported
separately.

## Repository

`https://github.com/chenzhuo2010/LCH-Seg-reproducibility`

## License

Code in `scripts/` is released under the MIT License (`LICENSE`). Result
tables, figure source data, fold assignments, configurations and documentation
are released under CC BY 4.0 (`LICENSE-DATA`). The source imaging datasets are
not covered by these licenses and keep the terms of their original releases.
