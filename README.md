# isles2022-25d-pipeline

**Language:** English | [Japanese](README_ja.md)

ISLES 2022 ischemic stroke lesion segmentation using a **2.5D ConvNeXt ensemble** — test mean Dice **0.631** (+22.8% relative vs 3D U-Net baseline).

**Quick links**
- English entry: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Japanese entry: [isles2022_25d/README.md](isles2022_25d/README.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)

## What this repository provides

- A 2.5D ConvNeXt-based pipeline for ISLES 2022 ischemic stroke lesion segmentation
- Two complementary model variants (v2: 5-slice, v3: 7-slice dilated) and a probability-map ensemble
- Tversky loss and EMA for improved small-lesion recall
- End-to-end workflow: train → single-model evaluate → ensemble evaluate
- Portfolio-ready documentation for external review
- A no-data smoke test that checks the public bundle in under a minute

## Who this is for

- Hiring managers reviewing medical AI segmentation work
- ML engineers looking for a lightweight 2.5D convolutional segmentation approach
- Researchers building on ISLES 2022 lesion segmentation baselines

## 3-minute overview

![2.5D ConvNeXt architecture](docs/assets/architecture.svg)

![Repository structure](docs/assets/repo_map.svg)

![Results snapshot](docs/assets/results_snapshot.svg)

### Results

| Model | val mean Dice | test mean Dice |
|-------|:---:|:---:|
| 3D U-Net baseline | 0.652 | 0.514 |
| ConvNeXt 2.5D v2 (5-slice) | 0.704 | ~0.58 |
| ConvNeXt 2.5D v3 (7-slice dilated) | 0.690 | 0.579 |
| **v2 + v3 ensemble** | **0.722** | **0.631** |

vs. 3D U-Net baseline: **test +0.117 (+22.8% relative)**

> Note: values come from local evaluation on the ISLES 2022 split. Protected medical data is not included in this repository.

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

- English guide: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- Japanese guide: [isles2022_25d/README.md](isles2022_25d/README.md)

## What is included vs excluded

Included:
- source code (models, datasets, training, evaluation)
- configs (v2 / v3 / vanilla 2.5D UNet)
- documentation and release notes
- static summary figures

Not included:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## Related

- **3D U-Net baseline**: [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline)

## How to cite

See [CITATION.cff](CITATION.cff).

## Commit message convention

Future commits follow Conventional Commits (`type: summary`):

- `fix: threshold default in eval script`
- `feat: add TTA support`
- `refactor: dataset loader`
- `docs: clarify v2 vs v3 config differences`
