# isles2022-25d-pipeline

ISLES-2022 虚血性脳卒中病変セグメンテーション — **2.5D ConvNeXt アプローチ**

**言語:** 日本語 | [English](README_en.md)

---

## 成果サマリ

| モデル | val mean Dice | test mean Dice |
|--------|:---:|:---:|
| 3D U-Net ベースライン | 0.652 | 0.514 |
| ConvNeXt 2.5D v2 (5-slice) | 0.704 | ~0.58 |
| ConvNeXt 2.5D v3 (7-slice dilated) | 0.690 | 0.579 |
| **v2 + v3 アンサンブル** | **0.722** | **0.631** |

3D U-Net 単体比: **test +0.117 (+22.8%)**

---

## アプローチ概要

### アーキテクチャ

隣接スライスをチャネル方向にスタックして **2D ConvNeXt エンコーダ** に入力する 2.5D 構成。
エンコーダの強力な表現学習能力と、3D 情報の軽量な取り込みを両立する。

```
入力: (B, C, H, W)
  C = n_slices × n_modalities
  例: 7 slices × 3 modalities (DWI+ADC+FLAIR) = 21ch

Encoder: ConvNeXt-Tiny (ImageNet pretrained)
  └─ first conv を multi-channel 入力に拡張 (repeat init)
  └─ 4 ステージ: 96ch → 192ch → 384ch → 768ch

Decoder: U-Net ライク skip-connection
  └─ lateral projection → ConvBlock × 3
  └─ Deep Supervision (1/4 + 1/8 スケール補助出力)

出力: (B, 1, H, W) — 中央スライスの病変確率マップ
```

### v2 vs v3 の違い

| | v2 | v3 |
|--|--|--|
| スライス数 | 5 (offsets: -2,-1,0,+1,+2) | 7 dilated (offsets: -5,-3,-1,0,+1,+3,+5) |
| 入力チャネル | 15ch | 21ch |
| Loss | Dice-OHEM-BCE | **Tversky(α=0.3,β=0.7)**-OHEM-BCE |
| EMA | なし | あり (decay=0.9998) |
| Epochs | 100 | 150 |

v3 は Tversky loss で小病変の recall を重視し、dilated offsets で広域文脈を取得。

### アンサンブル

v2 と v3 は入力スライス数・損失関数が異なるため確率マップの多様性があり、
単純平均アンサンブルで両者の弱点を補完できる。

---

## ファイル構成

```
core/pipeline/
├── configs/
│   ├── train_2_5d_unet.yaml               # vanilla 2.5D UNet (ベースライン)
│   ├── train_convnext_v2_5slice_1mm.yaml  # ConvNeXt v2
│   └── train_convnext_v3_7slice_dilated_1mm.yaml  # ConvNeXt v3
└── src/
    ├── datasets/
    │   └── isles_dataset.py               # IslesVolumeDataset / IslesSliceDataset
    ├── models/
    │   ├── convnext_nnunet_seg.py          # ConvNeXtNnUNetSeg (メインモデル)
    │   ├── input_adapters.py              # adapt_first_conv
    │   ├── unet_2_5d.py                  # vanilla 2.5D UNet (ベースライン)
    │   └── blocks_unet.py                # 共通 Conv ブロック
    ├── training/
    │   ├── train_isles_25d_convnext_fpn.py  # ConvNeXt 学習スクリプト
    │   ├── train_2_5d_unet.py            # vanilla 2.5D UNet 学習スクリプト
    │   ├── losses.py                     # Dice / Tversky / OHEM / Focal
    │   └── utils_train.py               # AverageMeter, EMA, sampler 等
    └── evaluation/
        ├── evaluate_isles_25d.py         # 単一モデル評価 (TTA 対応)
        └── evaluate_isles_25d_ensemble.py  # マルチモデルアンサンブル評価
```

---

## 再現手順

### 前提

- 前処理済みデータ (1mm 等方, DWI+ADC+FLAIR) が必要です。
  前処理スクリプトは [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline) を参照してください。
- `data/splits/isles2022_train_val_test.csv` (subject 列 + split 列) を用意してください。

### 1. v2 学習 (5-slice)

```bash
cd core/pipeline
PYTHONPATH=$PWD python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml
```

### 2. v3 学習 (7-slice dilated)

```bash
PYTHONPATH=$PWD python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v3_7slice_dilated_1mm.yaml
```

### 3. 単一モデル評価

```bash
PYTHONPATH=$PWD python -m src.evaluation.evaluate_isles_25d \
  --model-path results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_v3_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

### 4. アンサンブル評価 (v2 + v3)

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

## 設計上のポイント

| 課題 | 対応 |
|------|------|
| 小病変の見逃し | Tversky loss (β=0.7 で recall 重視) + OHEM |
| 過学習 | EMA, Dropout, pos_oversample sampler |
| val→test ギャップ | アンサンブルで分散低減 |
| 3D 情報の欠落 | Dilated slice offsets で広域文脈を確保 |

---

## 関連リポジトリ

- **3D U-Net ベースライン**: [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline)
