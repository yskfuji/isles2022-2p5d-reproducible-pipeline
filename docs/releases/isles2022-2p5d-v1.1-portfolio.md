# ISLES 2022 2.5D public snapshot — `isles2022-2p5d-v1.1-portfolio`

## Purpose
A portfolio snapshot that reflects the current public documentation, benchmark-summary tables, and wording cleanup.

## Snapshot type
Current portfolio release.

## Reviewer starting points
- README: repository overview, key metrics, and quickstart
- docs/reproducibility_checklist.md: external review checklist and pass criteria
- AUDIT_MAP.md: where major public artifacts and commands live

## Reproduced scope
- ISLES 2022 lesion segmentation with the 2.5D ConvNeXt pipeline
- Two related segmentation models and their probability-map ensemble
- Single-model and ensemble evaluation with TTA support
- Benchmark-summary tables for validation, local test, and lesion-size buckets

## Key results in this snapshot
- val mean Dice: **0.722** (ensemble), 0.704 (nearby-slice model), 0.690 (wide-context model)
- local-test mean Dice: **0.631** (bundled ensemble recipe)
- vs. 3D U-Net baseline (test 0.514): **+0.117 (+22.8% relative)**

## Main strengths in this snapshot
- Lightweight 2.5D design: 2D ConvNeXt encoder + multi-slice channel stacking
- Tversky loss (β=0.7) for improved small-lesion recall
- EMA for stable checkpointing
- Ensemble diversity from different slice spacings and loss functions

## Known limitations
- Protected medical data is not bundled
- GitHub Release text is mirrored from this source file
- Smoke test validates repository wiring, not full model quality