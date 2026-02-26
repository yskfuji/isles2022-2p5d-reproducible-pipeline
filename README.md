# isles2022-25d-pipeline

ISLES-2022 虚血性脳卒中病変セグメンテーション — **2.5D アプローチ**

## ブランチ構成

| ブランチ | 内容 |
|---|---|
| `main` | ベースライン: vanilla 2.5D UNet（隣接スライス Stack） |
| `dev` | 改善版: ConvNeXt-Tiny + nnU-Net ライク Decoder（開発中） |

---

## アプローチ概要

### `main` — Vanilla 2.5D UNet

隣接スライスをチャネル方向に Stack して 2D U-Net に入力する最小構成。
3D U-Net との比較ベースラインとして位置づけ。

- **入力**: DWI + ADC（`k_slices=2` → 5 スライス × 2 モダリティ = 10ch）
- **モデル**: `UNet2D`（Conv-BN-ReLU ×2 のエンコーダ/デコーダ、4 スケール）
- **損失**: Dice + BCE
- **解像度**: 256×256（1.5 mm 等方 resampling）

### `dev` — ConvNeXt-nnUNet（開発中）

- **Encoder**: ConvNeXt-Tiny（ImageNet pretrained, first-conv 拡張）
- **Decoder**: nnU-Net ライク skip-connection
- **改善点**: cosine LR warmup、Deep Supervision、aug 強化（affine/gamma/noise）
- **損失**: Dice-OHEM-BCE

---

## 依存関係

前処理・データセット・評価コード・環境設定は
**[isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline)**
を参照してください。

共有モジュール（本リポジトリでは非同梱）:
- `src/datasets/isles_dataset.py`
- `src/training/losses.py`
- `src/training/utils_train.py`

---

## ファイル構成

```
core/pipeline/
├── configs/
│   └── train_2_5d_unet.yaml      # 学習設定
└── src/
    ├── models/
    │   ├── blocks_unet.py         # Conv ブロック（共有ユーティリティ）
    │   └── unet_2_5d.py          # UNet2D モデル定義
    └── training/
        └── train_2_5d_unet.py    # 学習スクリプト
```

---

## 学習（main ブランチ）

```bash
# 3D repo の共有コードを同一環境に配置した上で実行
python -m src.training.train_2_5d_unet --config configs/train_2_5d_unet.yaml
```

---

## 関連リポジトリ

- **3D U-Net ベースライン**: [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline)
