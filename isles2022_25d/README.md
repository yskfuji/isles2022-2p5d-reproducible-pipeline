# ISLES-2022 — 再現性のある実験 README（ポートフォリオ向け）

**言語:** 日本語 | [英語版](README_en.md)

このフォルダは、**ISLES 2022 2.5D 病変セグメンテーション公開版の案内ページ**です。採用・監査・外部レビューの最初の接点として、価値・結果・最短の試し方を先に示す構成にしています。

## まず分かること

- **何ができるか**: 学習 / 単一モデル評価 / アンサンブル評価
- **誰向けか**: 採用担当、MRI セグメンテーション実装を見たい ML エンジニア、再現性重視の研究者
- **最短確認**: `python ../scripts/smoke_test.py --use_dummy_data`
- **成果の目安**:
  - ローカルテストの平均 Dice: **0.631**
  - アンサンブルの検証平均 Dice: **0.722**
  - 3D U-Net ベースラインのテスト Dice: **0.514**
  - 3D ベースライン比の改善幅: **+22.8%**

## すぐ使うリンク

- 英語版: [README_en.md](README_en.md)
- 詳細コード / 実験: `../core/pipeline/`
- 引用情報: `../CITATION.cff`
- リリースノート原稿: `../docs/releases/v1.0-interview.md`
- ロードマップ: `../ROADMAP.md`

## 内部ファイル名の対応表

| ファイル中の内部名 | 公開文書での意味 |
|---|---|
| `convnext_v2_5slice_1mm` | 近傍重視モデル（5 スライス） |
| `convnext_v3_7slice_dilated_1mm` | 広域文脈モデル（7 スライス・拡張間隔） |

## 固定スナップショット（ポートフォリオ用）

採用選考でレビューされた「再現評価」は、次のタグに対応します：

✅ `isles2022-2p5d-v1.0-interview`

リポジトリは継続的に開発中です。

対応する公開リポジトリ名: `isles2022-2p5d-reproducible-pipeline`

このフォルダは、ISLES-2022 病変セグメンテーションの 2.5D 実験を
**第三者が再現可能な形で理解し、実行できるようにするための案内ページ**です。

---

## TL;DR

- 主要コードは `../core/pipeline/` にあります。
- 2.5D ConvNeXt ベースの学習・評価・アンサンブル評価を一式で実行できます。
- 公開物には `Datasets/`・`runs/`・`results/` を同梱していません（データは各自で用意してください）。
- まずは「学習 → 評価 → アンサンブル評価」の最短 3 ステップを通すと全体像を把握できます。

## ベンチマーク要約

### この公開物で使っているアンサンブルの目安

| 対象 split | レシピ | Mean Dice | Global precision | Global recall | Lesion F1 |
|---|---|---:|---:|---:|---:|
| 検証 | Ensemble, `thr=0.85`, `min_size=32`, `prob_filter=0.96` | 0.721 | 0.854 | 0.815 | 0.501 |
| ローカルテスト | Ensemble, `thr=0.70`, `min_size=32`, `prob_filter=0.70` | 0.631 | 0.861 | 0.500 | 0.648 |

### アンサンブルのサイズ別 Dice

サイズ区分は evaluator の既定値に合わせています: small `<250` vox、medium `250-999` vox、large `>=1000` vox。

検証アンサンブル:

| GT 病変サイズ | 症例数 | Mean Dice | Median Dice | 検出率 | Lesion F1 |
|---|---:|---:|---:|---:|---:|
| Small | 4 | 0.3952 | 0.2903 | 0.25 | 0.1667 |
| Medium | 2 | 0.7005 | 0.7005 | 1.00 | 0.8333 |
| Large | 19 | 0.7924 | 0.8322 | 1.00 | 0.5370 |

ローカルテストのアンサンブル:

| GT 病変サイズ | 症例数 | Mean Dice | Median Dice | 検出率 | Lesion F1 |
|---|---:|---:|---:|---:|---:|
| Medium | 3 | 0.3652 | 0.3833 | 1.00 | 0.7333 |
| Large | 22 | 0.6674 | 0.7352 | 1.00 | 0.6360 |

注: ルート README からは、より厳密な fair-final 比較メモにもリンクしています。ここでは、この詳細 README で追いやすい公開レシピを表にまとめています。

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

> 注: 設定ファイル名や学習済み重みのパスには、内部実験名として旧来の `convnext_v2` / `convnext_v3` が残っています。公開文書上では、それぞれ **近傍重視モデル（5 スライス）**、**広域文脈モデル（7 スライス・拡張間隔）** として扱います。

### 2.1 近傍重視モデルの学習

```bash
python -m src.training.train_isles_25d_convnext_fpn \
  --config configs/train_convnext_v2_5slice_1mm.yaml
```

### 2.2 広域文脈モデルの学習

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
  --out-dir results/eval_7slice_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

### 2.4 アンサンブル評価

```bash
python -m src.evaluation.evaluate_isles_25d_ensemble \
  --model-paths results/convnext_v2_5slice_1mm/best.pt results/convnext_v3_7slice_dilated_1mm/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc_flair_1mm \
  --split test \
  --out-dir results/eval_ensemble_5slice_7slice_test \
  --thr 0.85 --min-size 32 --prob-filter 0.0
```

---

## 3. 現時点の要点（ポートフォリオ向け）

- 同系統の 2 モデルとして、近傍重視モデルと広域文脈モデルの設計差分、およびそのアンサンブル効果を検証しています。
- 小病変への対応として、Tversky loss、OHEM、EMA を組み合わせた設定を使っています。
- 既存レポートでは、テストで平均 Dice 0.631 のアンサンブル結果を確認しています（設定依存）。

---

## 4. 追加メモ

- 2.5D 版は 3D 前処理済みデータを前提にしています。
- 比較用 3D ベースラインは `isles2022-3d-reproducible-pipeline` を参照してください。

本公開物は `github_public_isles_25d/` 配下のみで参照が完結するように構成しています。
