# isles2022-2p5d-reproducible-pipeline

**Language:** English | [Japanese](README_ja.md)

Reproducible 2.5D ischemic stroke lesion segmentation pipeline for ISLES 2022, with audit-ready documentation, two related segmentation models, and ensemble-based evaluation.

**Quick links**
- English entry: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Japanese entry: [isles2022_25d/README.md](isles2022_25d/README.md)
- Detailed documentation: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Reproducibility checklist: [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)

## What this repository provides

- A reproducible train-evaluate-ensemble workflow for ISLES 2022 lesion segmentation
- Two related segmentation models: a nearby-slice model and a wide-context model
- Ensemble evaluation for a lightweight alternative to full 3D segmentation
- Richer evaluation utilities including threshold sweep, postprocess sweep, probability-map reuse, temperature fitting, and extra non-Dice metrics
- Portfolio-ready documentation for external review
- A no-data smoke test that checks the public bundle in under a minute

### Metric snapshot beyond Dice

Final fair test recipe:
- validation-selected threshold: `0.75`
- `min_size=32`
- `prob_filter=0.90`

Final fair test metrics for the ensemble:

| Metric | Value |
|---|---:|
| Mean Dice | 0.621 |
| Voxel precision | 0.856 |
| Voxel recall | 0.539 |
| Lesion-wise micro F1 | 0.536 |
| ASSD | 6.89 mm |
| HD95 | 21.40 mm |
| Mean absolute volume error | 15.95 mL |

Compact comparison:

| Split | Model | Dice | Precision | Recall | Lesion F1 |
|---|---|---:|---:|---:|---:|
| Validation | Nearby-slice model | 0.704 | 0.859 | 0.813 | 0.513 |
| Validation | Wide-context model | 0.713 | 0.830 | 0.827 | 0.543 |
| Validation | Ensemble | 0.722 | 0.803 | 0.865 | 0.615 |
| Test | Nearby-slice model | 0.588 | 0.874 | 0.449 | 0.416 |
| Test | Wide-context model | 0.624 | 0.892 | 0.485 | 0.471 |
| Test | Ensemble (fair final) | 0.621 | 0.856 | 0.539 | 0.536 |

Full validation / test comparison note: [artifacts/eval_runs/metrics_comparison_20260311.md](artifacts/eval_runs/metrics_comparison_20260311.md)

## Internal filename mapping

Some config files and checkpoint examples still preserve historical internal experiment names. Both models share the same overall design: a ConvNeXt encoder with a U-Net-style decoder.

| Internal name in files | Public-facing meaning |
|---|---|
| `convnext_v2_5slice_1mm` | Nearby-slice model (5 slices) |
| `convnext_v3_7slice_dilated_1mm` | Wide-context model (7-slice dilated) |

## Who this is for

- Hiring managers reviewing medical AI segmentation work
- ML engineers looking for an auditable MRI segmentation baseline
- Researchers looking for a reproducible ISLES-style 2.5D project structure

## 3-minute overview

![ISLES 2.5D architecture](docs/assets/architecture.svg)

![ISLES 2.5D repository map](docs/assets/repo_map.svg)

![ISLES 2.5D metrics snapshot](docs/assets/results_snapshot.svg)

### Representative results

| Metric | Value | Why it matters |
|---|---:|---|
| Local test mean Dice | 0.631 | Practical performance snapshot for the bundled 2.5D ensemble recipe |
| Validation mean Dice | 0.722 | Demonstrates stronger in-distribution validation behavior for the two-model ensemble |
| Local test lesion-wise F1 | 0.536 | Indicates stronger lesion-level detection quality than Dice alone |
| Local test voxel recall | 0.539 | Shows the ensemble trades some precision for better lesion coverage |
| 3D U-Net baseline test Dice | 0.514 | Provides a local comparison anchor |
| Relative gain vs 3D baseline | +22.8% | Shows the benefit of the 2.5D ensemble in the local test setting |

> Notes: values are configuration-dependent and come from the bundled recipe / evaluation notes. Protected medical data is intentionally not included.

## Quickstart

### 1. Verify the repository without medical data

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. Inspect the public bundle manifest

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. Run full training / evaluation with your own data

- Full guide in English: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Full guide in Japanese: [isles2022_25d/README.md](isles2022_25d/README.md)

## What is included vs excluded

Included:
- source code
- configs
- audit and evaluation documentation
- static summary figures and release-note sources

Not included:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## Stable portfolio version

Active development continues in this repository. The stable snapshot used for portfolio and interview review is:

✅ `isles2022-2p5d-v1.0-interview`

## How to cite

See [CITATION.cff](CITATION.cff).

## Commit message convention

To keep ongoing changes reviewable, future commits follow Conventional Commits (`type: summary`):

- `fix: threshold default in eval script`
- `feat: add ensemble calibration notes`
- `refactor: slice sampler validation`
- `docs: clarify 5-slice vs 7-slice evaluation protocol`
