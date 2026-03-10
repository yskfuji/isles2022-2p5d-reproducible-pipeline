# isles2022-2p5d-reproducible-pipeline

**Language:** English | [Japanese](README_ja.md)

Reproducible 2.5D ischemic stroke lesion segmentation pipeline for ISLES 2022, with audit-ready documentation, multi-slice ConvNeXt variants, and ensemble-based evaluation.

**Quick links**
- English entry: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Japanese entry: [isles2022_25d/README.md](isles2022_25d/README.md)
- Detailed documentation: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)

## What this repository provides

- A reproducible train-evaluate-ensemble workflow for ISLES 2022 lesion segmentation
- 2.5D ConvNeXt baselines with explicit differences between the 5-slice model and the 7-slice dilated model
- Ensemble evaluation for a lightweight alternative to full 3D segmentation
- Portfolio-ready documentation for external review
- A no-data smoke test that checks the public bundle in under a minute

## Internal filename mapping

Some config files and checkpoint examples still preserve historical internal experiment names. In public-facing documents, use the following mapping:

| Internal name in files | Public-facing meaning |
|---|---|
| `convnext_v2_5slice_1mm` | 5-slice model |
| `convnext_v3_7slice_dilated_1mm` | 7-slice dilated model |

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
| Validation mean Dice | 0.722 | Demonstrates stronger in-distribution validation behavior for the ensemble |
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
