# isles2022-25d-pipeline

ISLES-2022 Ischemic Stroke Lesion Segmentation — **2.5D ConvNeXt Approach**

**Language:** [Japanese](README.md) | English

---

## Results Summary

| Model | val mean Dice | test mean Dice |
|-------|:---:|:---:|
| 3D U-Net baseline | 0.652 | 0.514 |
| ConvNeXt 2.5D v2 (5-slice) | 0.704 | ~0.58 |
| ConvNeXt 2.5D v3 (7-slice dilated) | 0.690 | 0.579 |
| **v2 + v3 ensemble** | **0.722** | **0.631** |

vs. 3D U-Net baseline: **test +0.117 (+22.8% relative)**

---

## Approach

### Architecture

Adjacent MRI slices are stacked along the channel dimension and fed into a **2D ConvNeXt encoder**, giving a lightweight 2.5D design that captures inter-slice context without the memory cost of full 3D convolutions.

```
Input: (B, C, H, W)
  C = n_slices × n_modalities
  e.g. 7 slices × 3 modalities (DWI+ADC+FLAIR) = 21ch

Encoder: ConvNeXt-Tiny (ImageNet pretrained)
  └─ first conv extended to multi-channel input (repeat init)
  └─ 4 stages: 96 → 192 → 384 → 768 channels

Decoder: U-Net skip-connections
  └─ lateral projections + ConvBlock × 3
  └─ Deep Supervision (1/4 and 1/8 scale auxiliary outputs)

Output: (B, 1, H, W) — lesion probability map for the center slice
```

### v2 vs v3

| | v2 | v3 |
|--|--|--|
| Slices | 5 (offsets: -2,-1,0,+1,+2) | 7 dilated (offsets: -5,-3,-1,0,+1,+3,+5) |
| Input channels | 15ch | 21ch |
| Loss | Dice-OHEM-BCE | **Tversky(α=0.3,β=0.7)**-OHEM-BCE |
| EMA | — | yes (decay=0.9998) |
| Epochs | 100 | 150 |

v3 uses dilated offsets for wider context and Tversky loss to emphasise recall for small lesions.

### Ensemble

v2 and v3 differ in slice spacing and loss function, introducing prediction diversity. A simple probability-map average ensemble complements each model's weaknesses.

---

## File Structure

```
core/pipeline/
├── configs/
│   ├── train_2_5d_unet.yaml               # vanilla 2.5D UNet (baseline)
│   ├── train_convnext_v2_5slice_1mm.yaml  # ConvNeXt v2
│   └── train_convnext_v3_7slice_dilated_1mm.yaml  # ConvNeXt v3
└── src/
    ├── datasets/
    │   └── isles_dataset.py               # IslesVolumeDataset / IslesSliceDataset
    ├── models/
    │   ├── convnext_nnunet_seg.py          # ConvNeXtNnUNetSeg (main model)
    │   ├── input_adapters.py              # adapt_first_conv
    │   ├── unet_2_5d.py                  # vanilla 2.5D UNet (baseline)
    │   └── blocks_unet.py                # shared Conv blocks
    ├── training/
    │   ├── train_isles_25d_convnext_fpn.py  # ConvNeXt training script
    │   ├── train_2_5d_unet.py            # vanilla 2.5D UNet training
    │   ├── losses.py                     # Dice / Tversky / OHEM / Focal losses
    │   └── utils_train.py               # AverageMeter, EMA, sampler, etc.
    └── evaluation/
        ├── evaluate_isles_25d.py         # single-model evaluation (TTA support)
        └── evaluate_isles_25d_ensemble.py  # multi-model ensemble evaluation
```

---

## Reproduction

### Prerequisites

- Preprocessed 1 mm isotropic DWI+ADC+FLAIR volumes are required.
  See [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline) for preprocessing.
- Prepare `data/splits/isles2022_train_val_test.csv` with `subject` and `split` columns.

### 1. Train v2 (5-slice)

```bash
cd core/pipeline
PYTHONPATH=$PWD python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml
```

### 2. Train v3 (7-slice dilated)

```bash
PYTHONPATH=$PWD python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v3_7slice_dilated_1mm.yaml
```

### 3. Evaluate single model

```bash
PYTHONPATH=$PWD python -m src.evaluation.evaluate_isles_25d \
  --model-path results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_v3_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

### 4. Ensemble evaluation (v2 + v3)

```bash
PYTHONPATH=$PWD python -m src.evaluation.evaluate_isles_25d_ensemble \
  --model-paths \
    results/convnext_v2_5slice_1mm/best.pt \
    results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_ens_v2_v3_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

---

## Design Notes

| Challenge | Solution |
|-----------|----------|
| Small-lesion recall | Tversky loss (β=0.7) + OHEM |
| Overfitting | EMA, Dropout, positive-slice oversampling |
| val→test gap | Ensemble reduces variance |
| Limited 3D context | Dilated slice offsets capture wider range |

---

## Related

- **3D U-Net baseline**: [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline)
