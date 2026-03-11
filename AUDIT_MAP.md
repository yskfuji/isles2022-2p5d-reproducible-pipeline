# ISLES 2.5D 監査マップ

この公開物は ISLES 2.5D 監査向けに最短導線で整理しています。

## 1. 読む順番

1. `./isles2022_25d/README.md`（または `README_en.md`）
2. `./core/pipeline/configs/train_convnext_v2_5slice_1mm.yaml`
3. `./core/pipeline/configs/train_convnext_v3_7slice_dilated_1mm.yaml`
4. `./core/pipeline/src/models/convnext_nnunet_seg.py`
5. `./core/pipeline/src/training/train_isles_25d_convnext_fpn.py`
6. `./core/pipeline/src/evaluation/evaluate_isles_25d.py`
7. `./core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py`

## 2. 主な監査ポイント

- 2.5D 設計: 隣接スライスをチャネル方向にスタックして 2D ConvNeXt に入力
- 近傍重視モデル vs 広域文脈モデル: スライス間隔・損失関数・EMA の違いによる多様性
- アンサンブル: 確率マップ単純平均で分散を低減
- 小病変対策: Tversky loss (β=0.7 で recall 重視) + OHEM

## 3. 内部ファイル名の対応

- `train_convnext_v2_5slice_1mm.yaml` → 近傍重視モデル（5 スライス）
- `train_convnext_v3_7slice_dilated_1mm.yaml` → 広域文脈モデル（7 スライス dilated）

## 4. 除外物

- `Datasets/`
- `runs/`
- `results/`
- `logs/`
