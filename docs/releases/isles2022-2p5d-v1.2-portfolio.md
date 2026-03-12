# ISLES 2022 2.5D public snapshot — `isles2022-2p5d-v1.2-portfolio`

## Purpose
A portfolio snapshot that extends the public 2.5D pipeline from experiment tracking to registry-ready model promotion.

## Snapshot type
Current portfolio release.

## Reviewer starting points
- README: repository overview, key metrics, and quickstart
- docs/reproducibility_checklist.md: external review checklist and pass criteria
- AUDIT_MAP.md: where major public artifacts and commands live
- core/pipeline/tools/register_model.py: registry-ready bundle creation and promotion-rule entrypoint

## Reproduced scope
- ISLES 2022 lesion segmentation with the 2.5D ConvNeXt pipeline
- Two related segmentation models and their probability-map ensemble
- Shared MLflow tracking schema across the public portfolio repositories
- Registry-ready model bundle creation from completed runs
- Optional MLflow Registry handoff with promotion-rule evaluation and alias updates

## Key results in this snapshot
- val mean Dice: **0.722** (ensemble), 0.704 (nearby-slice model), 0.690 (wide-context model)
- local-test mean Dice: **0.631** (bundled ensemble recipe)
- vs. 3D U-Net baseline (test 0.514): **+0.117 (+22.8% relative)**

## Main strengths in this snapshot
- Lightweight 2.5D design: 2D ConvNeXt encoder + multi-slice channel stacking
- Shared run metadata and artifact layout across the portfolio repositories
- Configurable promotion rules based on logged validation metrics
- Optional MLflow Registry alias updates for candidate or champion handoff
- Portfolio-visible MLOps progression without claiming automated deployment

## Known limitations
- Protected medical data is not bundled
- GitHub Release text is mirrored from this source file
- Smoke test validates repository wiring, not full model quality
- Promotion rules evaluate the latest logged metrics and still require human-controlled thresholds