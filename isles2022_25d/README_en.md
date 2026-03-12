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
- Release-note source: `../docs/releases/isles2022-2p5d-v1.2-portfolio.md`
- Roadmap: `../ROADMAP.md`

## Internal filename mapping

| Internal name in files | Public-facing meaning |
|---|---|
| `convnext_v2_5slice_1mm` | Nearby-slice model (5 slices) |
| `convnext_v3_7slice_dilated_1mm` | Wide-context model (7-slice dilated) |

## Current Portfolio Snapshot

The current portfolio snapshot corresponds to:

✅ `isles2022-2p5d-v1.2-portfolio`

Active development continues on the repository.

This folder is the public entry point for ISLES-2022 2.5D lesion-segmentation work,
organized so a third party can understand and rerun the pipeline with their own data.

---

## TL;DR

- Main implementation lives in `../core/pipeline/`.
- The repository includes end-to-end scripts for training, single-model evaluation, and ensemble evaluation.
- `Datasets/`, `runs/`, and `results/` are intentionally excluded from this public export.
- The fastest way to understand the project is: train → evaluate → ensemble.

## Benchmark summary

### Ensemble snapshot used in the public bundle

| Split | Recipe | Mean Dice | Global precision | Global recall | Lesion F1 |
|---|---|---:|---:|---:|---:|
| Validation | Ensemble, `thr=0.85`, `min_size=32`, `prob_filter=0.96` | 0.721 | 0.854 | 0.815 | 0.501 |
| Local test | Ensemble, `thr=0.70`, `min_size=32`, `prob_filter=0.70` | 0.631 | 0.861 | 0.500 | 0.648 |

### Size-stratified ensemble Dice

Buckets follow the evaluator default: small `<250` vox, medium `250-999` vox, large `>=1000` vox.

Validation ensemble recipe:

| GT lesion size | Cases | Mean Dice | Median Dice | Detection rate | Lesion F1 |
|---|---:|---:|---:|---:|---:|
| Small | 4 | 0.3952 | 0.2903 | 0.25 | 0.1667 |
| Medium | 2 | 0.7005 | 0.7005 | 1.00 | 0.8333 |
| Large | 19 | 0.7924 | 0.8322 | 1.00 | 0.5370 |

Local-test ensemble recipe:

| GT lesion size | Cases | Mean Dice | Median Dice | Detection rate | Lesion F1 |
|---|---:|---:|---:|---:|---:|
| Medium | 3 | 0.3652 | 0.3833 | 1.00 | 0.7333 |
| Large | 22 | 0.6674 | 0.7352 | 1.00 | 0.6360 |

Note: the root README links a stricter fair-final comparison note separately; the tables above summarize the bundled portfolio recipes used in this detailed entry page.

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

> Note: some config filenames and checkpoint paths still use the historical internal labels `convnext_v2` and `convnext_v3`. In public-facing documentation, these are described as the **nearby-slice model (5 slices)** and the **wide-context model (7-slice dilated)**.

### 2.1 Train the nearby-slice model

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml
```

Optional MLflow tracking:

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml \
  --mlflow --mlflow-experiment isles-25d-convnext
```

If `mlflow` is not installed in your local environment, install it first and rerun with the same command.

### 2.2 Train the wide-context model

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

## 3. MLflow tracking schema

When `--mlflow` is enabled, this repository follows the same public tracking schema used across the three portfolio repositories.

- Common run tags: `repo_name`, `task_type`, `model_family`, `tracking_schema=public_portfolio_v1`
- Common artifact groups:
  - `run_metadata/`: `meta.json` and, when available, a config snapshot or task-specific JSON
  - `training_trace/`: `log.jsonl`
  - `checkpoints/`: `last.pt`, `best.pt`, and any task-specific best-checkpoint variants
- Goal: make run review easier across segmentation and classification repos without claiming a production MLOps platform

## 4. Model registration stage

The next MLOps step in this public portfolio is a registry-ready bundle created from a completed training run.

```bash
python tools/register_model.py \
  --run-dir runs/convnext_v3_7slice_dilated_1mm/<YOUR_RUN> \
  --model-name isles-25d-convnext \
  --version-label ensemble-candidate \
  --checkpoint best.pt \
  --selection-reason "candidate for 2.5D ensemble promotion" \
  --promotion-rule "val_dice>=0.72"
```

This command creates `artifacts/registered_models/<model-name>/<version-label>/` and stores:

- `registration.json`
- `run_metadata/`
- `training_trace/`
- `checkpoints/`

Optional MLflow handoff:

```bash
python tools/register_model.py \
  --run-dir runs/convnext_v3_7slice_dilated_1mm/<YOUR_RUN> \
  --model-name isles-25d-convnext \
  --version-label ensemble-candidate \
  --checkpoint best.pt \
  --promotion-rule "val_dice>=0.72" \
  --mlflow-register \
  --mlflow-experiment isles-model-registration \
  --registered-model-name isles-25d-convnext \
  --promote-alias candidate \
  --reject-alias needs-review
```

Promotion rules are evaluated against the latest metrics in `log.jsonl`. If the rule passes, the configured promotion alias is updated on the created MLflow model version.

If you want the verification flow in a single command, use `verify_registration.py`.

```bash
python tools/verify_registration.py \
  --run-dir runs/convnext_v3_7slice_dilated_1mm/<YOUR_RUN> \
  --model-name isles-25d-convnext \
  --version-label verify-ensemble-candidate \
  --checkpoint best.pt \
  --promotion-rule "val_dice>=0.72" \
  --registered-model-name isles-25d-convnext-verify
```

This wrapper creates a local SQLite-backed MLflow Registry under `artifacts/verification/`, runs `register_model.py`, and prints a compact JSON summary that confirms both `registration.json` and the expected alias.

---

## 5. Current highlights (portfolio notes)

- The pipeline centers on two related segmentation models, comparing a nearby-slice model with a wide-context model and their ensemble effect.
- Small-lesion handling is supported through settings such as Tversky loss, OHEM, and EMA.
- Existing reports include ensemble runs with mean Dice around 0.631 on the local test setting, depending on configuration.


---

## 6. Additional notes

- The 2.5D setup assumes preprocessed volumes prepared through the 3D pipeline.
- For the comparison baseline, see `isles2022-3d-reproducible-pipeline`.

This public package is self-contained under `github_public_isles_25d/`.
