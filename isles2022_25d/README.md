# ISLES-2022 — 再現性のある実験 README（ポートフォリオ向け）

**言語:** 日本語 | [英語版](README_en.md)

このフォルダは、**ISLES 2022 2.5D 病変セグメンテーション公開版の入口**です。採用・監査・外部レビューの最初の接点として、価値・結果・最短の試し方を先に示す構成にしています。

## まず分かること

- **何ができるか**: 学習 / 単一モデル評価 / アンサンブル評価
- **誰向けか**: 採用担当、MRI セグメンテーション実装を見たい ML エンジニア、再現性重視の研究者
- **最短確認**: `python ../scripts/smoke_test.py --use_dummy_data`
- **成果の目安**:
  - local test mean Dice: **0.631**
  - ensemble validation mean Dice: **0.722**
  - 3D U-Net baseline test Dice: **0.514**
  - relative gain vs 3D baseline: **+22.8%**

## すぐ使うリンク

- 英語版: [README_en.md](README_en.md)
- 詳細コード / 実験: `../core/pipeline/`
- 引用情報: `../CITATION.cff`
- リリースノート原稿: `../docs/releases/v1.0-interview.md`
- ロードマップ: `../ROADMAP.md`

## 固定スナップショット（ポートフォリオ用）

採用選考でレビューされた「再現評価」は、次のタグに対応します：

✅ `isles2022-2p5d-v1.0-interview`

リポジトリは継続的に開発中です。

対応する公開リポジトリ名: `isles2022-2p5d-reproducible-pipeline`

このフォルダは、ISLES-2022 病変セグメンテーションの 2.5D 実験を
**第三者が再現可能な形で理解し、実行できるようにするための入口**です。

---

## TL;DR

- 主要コードは `../core/pipeline/` にあります。
- 2.5D ConvNeXt ベースの学習・評価・アンサンブル評価を一式で実行できます。
- 公開物には `Datasets/`・`runs/`・`results/` を同梱していません（データは各自で用意してください）。
- まずは「学習 → 評価 → アンサンブル評価」の最短 3 ステップを通すと全体像を把握できます。

---

## 1. コードマップ

- 学習（2.5D ConvNeXt）
  - `../core/pipeline/src/training/train_isles_25d_convnext_fpn.py`
- 学習（2.5D U-Net ベースライン）
  - `../core/pipeline/src/training/train_2_5d_unet.py`
- 評価（単一モデル / TTA 対応）
  - `../core/pipeline/src/evaluation/evaluate_isles_25d.py`
- 評価（アンサンブル）
  - `../core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py`
- データセット定義
  - `../core/pipeline/src/datasets/isles_dataset.py`

---

## 2. 再現手順（最短）

以下は `github_public_isles_25d/core/pipeline/` をカレントとして実行します。

### 2.1 v2 学習

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml
```

### 2.2 v3 学習

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v3_7slice_dilated_1mm.yaml
```

### 2.3 評価

```bash
python -m src.evaluation.evaluate_isles_25d \
  --model-path results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_v3_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

### 2.4 アンサンブル評価

```bash
python -m src.evaluation.evaluate_isles_25d_ensemble \
  --model-paths results/convnext_v2_5slice_1mm/best.pt results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_ens_v2_v3_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

---

## 3. 現時点の要点（ポートフォリオ向け）

- 2.5D ConvNeXt を主軸に、v2 / v3 のスライス設計差分とアンサンブル効果を検証しています。
- 小病変への対応として、Tversky loss、OHEM、EMA を組み合わせた設定を使っています。
- 既存レポートでは、test で mean Dice 0.631 のアンサンブル結果を確認しています（設定依存）。

---

## 4. 追加メモ

- 2.5D 版は 3D 前処理済みデータを前提にしています。
- 比較用 3D ベースラインは `isles2022-3d-reproducible-pipeline` を参照してください。

本公開物は `github_public_isles_25d/` 配下のみで参照が完結するように構成しています。
