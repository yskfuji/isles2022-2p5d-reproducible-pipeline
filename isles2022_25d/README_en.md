# ISLES-2022 — Reproducible Experiment README (Portfolio)

**Language:** English | [Japanese](README.md)

This folder is the **public entry point** for the ISLES 2022 2.5D lesion-segmentation pipeline. It is designed so an external reviewer can understand the project value, representative results, and the fastest way to try it before reading the full experiment notes.

## What a reviewer can verify quickly

- **What it does**: train / evaluate / ensemble a 2.5D ConvNeXt-based ISLES pipeline
- **Who it is for**: hiring managers, ML engineers, and researchers who want reproducible MRI segmentation work
- **Fastest first run**: `python ../scripts/smoke_test.py --use_dummy_data`
- **Representative metrics**:
  - local test mean Dice: **0.631**
  - ensemble validation mean Dice: **0.722**
  - 3D U-Net baseline test Dice: **0.514**
  - relative gain vs 3D baseline: **+22.8%**

## Quick links

- Japanese version: [README.md](README.md)
- Detailed code / experiments: `../core/pipeline/`
- Citation: `../CITATION.cff`
- Release-note source: `../docs/releases/v1.0-interview.md`
- Roadmap: `../ROADMAP.md`

## Stable Portfolio Version

The reproducible evaluation reviewed during recruitment corresponds to:

✅ `isles2022-2p5d-v1.0-interview`

Active development continues on the repository.

This folder is the public entry point for ISLES-2022 2.5D lesion-segmentation work,
organized so a third party can understand and rerun the pipeline with their own data.

---

## TL;DR

- Main implementation lives in `../core/pipeline/`.
- The repository includes end-to-end scripts for training, single-model evaluation, and ensemble evaluation.
- `Datasets/`, `runs/`, and `results/` are intentionally excluded from this public export.
- The fastest way to understand the project is: train → evaluate → ensemble.

---

## 1. Code map

- Training (2.5D ConvNeXt)
  - `../core/pipeline/src/training/train_isles_25d_convnext_fpn.py`
- Training (2.5D U-Net baseline)
  - `../core/pipeline/src/training/train_2_5d_unet.py`
- Evaluation (single model / TTA)
  - `../core/pipeline/src/evaluation/evaluate_isles_25d.py`
- Evaluation (ensemble)
  - `../core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py`
- Dataset definitions
  - `../core/pipeline/src/datasets/isles_dataset.py`

---

## 2. Quick reproducible run

Run the commands below from `github_public_isles_25d/core/pipeline/`.

> Note: some config filenames and checkpoint paths still use the historical internal labels `convnext_v2` and `convnext_v3`. In public-facing documentation, these are described as the **5-slice model** and the **7-slice dilated model**.

### 2.1 Train the 5-slice model

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml
```

### 2.2 Train the 7-slice dilated model

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v3_7slice_dilated_1mm.yaml
```

### 2.3 Evaluate

```bash
python -m src.evaluation.evaluate_isles_25d \
  --model-path results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_7slice_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

### 2.4 Ensemble evaluation

```bash
python -m src.evaluation.evaluate_isles_25d_ensemble \
  --model-paths results/convnext_v2_5slice_1mm/best.pt results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_ensemble_5slice_7slice_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

---

## 3. Current highlights (portfolio notes)

- The pipeline centers on two 2.5D ConvNeXt variants, comparing a 5-slice design with a 7-slice dilated design and their ensemble effect.
- Small-lesion handling is supported through settings such as Tversky loss, OHEM, and EMA.
- Existing reports include ensemble runs with mean Dice around 0.631 on the local test setting, depending on configuration.

---

## 4. Additional notes

- The 2.5D setup assumes preprocessed volumes prepared through the 3D pipeline.
- For the comparison baseline, see `isles2022-3d-reproducible-pipeline`.

This public package is self-contained under `github_public_isles_25d/`.
