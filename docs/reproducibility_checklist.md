# Reproducibility Checklist

Use this page as a fast external-review checklist for the public ISLES 2022 2.5D repository.

## 1. Package integrity

- Confirm the stable snapshot tag referenced in the README and release note.
- Confirm the repository excludes protected medical data and private training artifacts.
- Optionally generate a fresh manifest with `python scripts/smoke_test.py --use_dummy_data` or `python tools/make_manifest.py` from `core/pipeline`.

## 2. Documentation consistency

- Read the landing page: `README.md` or `README_ja.md`.
- Read the task-facing guide: `isles2022_25d/README_en.md` or `isles2022_25d/README.md`.
- Read the release note source under `docs/releases/v1.0-interview.md`.
- Confirm the reported ensemble metrics and model naming are consistent across those files.

## 3. Code-path sanity

- Verify that training and evaluation entrypoints exist:
  - `core/pipeline/src/training/train_isles_25d_convnext_fpn.py`
  - `core/pipeline/src/evaluation/evaluate_isles_25d.py`
  - `core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py`
- Verify that threshold sweep, postprocess sweep, probability-map reuse, and temperature fitting are present in the public tools.

## 4. Smoke-test validation

- Run `python scripts/smoke_test.py --use_dummy_data`.
- Confirm the command completes without requiring medical data.
- Confirm the generated summary points at the expected public files and entrypoints.

## 5. Evaluation-readiness checks

- Confirm the README reports both Dice and non-Dice metrics.
- Confirm the final ensemble recipe is explained with threshold and postprocess settings.
- Confirm the repository explains the mapping between internal experiment names and public-facing names.

## 6. Reviewer pass criteria

- A reviewer can understand the two-model setup and the ensemble role in under 3 minutes.
- A reviewer can trace train -> evaluate -> ensemble without needing hidden scripts.
- A reviewer can validate repository wiring without access to protected data.

## 7. Known limits

- This checklist validates public reproducibility scaffolding, not end-to-end training on real ISLES data.
- Full metric reproduction still requires separately prepared ISLES 2022 data and checkpoints.