# isles2022-2p5d-reproducible-pipeline

**言語:** 日本語 | [English](README.md)

ISLES 2022 向けの、**再現可能な 2.5D 脳梗塞病変セグメンテーションパイプライン**です。監査しやすいドキュメント、同系統の 2 モデル、アンサンブル評価を含みます。

**クイックリンク**
- 英語版: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- 日本語版: [isles2022_25d/README.md](isles2022_25d/README.md)
- 実験詳細: [isles2022_25d/README.md](isles2022_25d/README.md)
- 再現性チェックリスト: [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md)
- 引用情報: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)

## このリポジトリでできること

- ISLES 2022 病変セグメンテーションの学習 → 評価 → アンサンブル評価ワークフロー
- 2 つのセグメンテーションモデルとして、近傍重視モデルと広域文脈モデルの設計差分を比較
- フル 3D より軽量な構成での病変セグメンテーション実験
- 閾値 sweep、後処理 sweep、probability map 再利用、温度スケーリング、Dice 以外の追加指標を含む評価ツール群
- 外部レビュー向けに整理したポートフォリオ導線
- 実データなしで公開物の配線を確認できる簡易動作確認

### Dice 以外も含む指標スナップショット

公平な最終 test 条件:
- validation で選んだ threshold: `0.75`
- `min_size=32`
- `prob_filter=0.90`

最終 ensemble の test 指標:

| 指標 | 値 |
|---|---:|
| Mean Dice | 0.621 |
| Voxel precision | 0.856 |
| Voxel recall | 0.539 |
| Lesion-wise micro F1 | 0.536 |
| ASSD | 6.89 mm |
| HD95 | 21.40 mm |
| Mean absolute volume error | 15.95 mL |

比較サマリー:

| Split | モデル | Dice | Precision | Recall | Lesion F1 |
|---|---|---:|---:|---:|---:|
| Validation | 近傍重視モデル | 0.704 | 0.859 | 0.813 | 0.513 |
| Validation | 広域文脈モデル | 0.713 | 0.830 | 0.827 | 0.543 |
| Validation | Ensemble | 0.722 | 0.803 | 0.865 | 0.615 |
| Test | 近傍重視モデル | 0.588 | 0.874 | 0.449 | 0.416 |
| Test | 広域文脈モデル | 0.624 | 0.892 | 0.485 | 0.471 |
| Test | Ensemble（公平な最終値） | 0.621 | 0.856 | 0.539 | 0.536 |

詳細な validation / test 比較表: [artifacts/eval_runs/metrics_comparison_20260311.md](artifacts/eval_runs/metrics_comparison_20260311.md)

## 内部ファイル名の対応表

設定ファイル名や学習済み重みの例では、過去の内部実験名が残っています。両者とも基本構成は同じで、ConvNeXt encoder と U-Net 風 decoder を使います。

| ファイル中の内部名 | 公開文書での意味 |
|---|---|
| `convnext_v2_5slice_1mm` | 近傍重視モデル（5 スライス） |
| `convnext_v3_7slice_dilated_1mm` | 広域文脈モデル（7 スライス dilated） |

## 想定している読者

- 医療AIセグメンテーション実装を確認したい採用担当
- 監査しやすい MRI セグメンテーション基盤を見たい ML エンジニア
- 再現性重視の ISLES 系 2.5D プロジェクト構成を探している研究者

## 3分で分かる概要

![ISLES 2.5D パイプライン構成図](docs/assets/architecture.svg)

![ISLES 2.5D リポジトリ構成図](docs/assets/repo_map.svg)

![ISLES 2.5D 指標サマリー](docs/assets/results_snapshot.svg)

### 代表指標

| 指標 | 値 | 意味 |
|---|---:|---|
| Local test mean Dice | 0.631 | 公開レシピの実用的な性能目安 |
| Validation mean Dice | 0.722 | 2 モデルを組み合わせたアンサンブルの検証性能 |
| Local test lesion-wise F1 | 0.536 | Dice だけでは見えにくい病変単位の検出性能 |
| Local test voxel recall | 0.539 | 最終 ensemble の病変取りこぼしの少なさを補足 |
| 3D U-Net baseline test Dice | 0.514 | 局所比較の基準値 |
| Relative gain vs 3D baseline | +22.8% | 2.5D アンサンブルによる改善幅 |

> 数値は同梱レシピと評価メモに基づきます。医療データ本体は公開物に含めていません。

## 最短の確認方法

### 1. 実データなしで配線確認

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. 配布物マニフェストを確認

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. 実データで学習 / 評価

- 日本語詳細: [isles2022_25d/README.md](isles2022_25d/README.md)
- 英語版の詳細ガイド: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)

## 含まれるものと含まれないもの

含まれるもの:
- ソースコード
- 設定ファイル
- 監査 / 評価ドキュメント
- 静的図表と release note 原稿

含まれないもの:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## 固定スナップショット（ポートフォリオ用）

開発は継続中ですが、ポートフォリオ / 面接レビュー用の固定版は次のタグです。

✅ `isles2022-2p5d-v1.0-interview`

## 引用

[CITATION.cff](CITATION.cff) を参照してください。

## コミットメッセージの規約

今後の変更は Conventional Commits（`type: summary`）で揃えます。

- `fix: eval スクリプトの閾値デフォルト値修正`
- `feat: add ensemble calibration notes`
- `refactor: slice sampler validation`
- `docs: clarify 5-slice vs 7-slice evaluation protocol`
