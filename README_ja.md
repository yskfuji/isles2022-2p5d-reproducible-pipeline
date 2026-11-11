# isles2022-2p5d-reproducible-pipeline

**言語:** 日本語 | [English](README.md)

本リポジトリは、ISLES 2022 向けの再現可能な 2.5D 脳梗塞病変セグメンテーションパイプラインを公開する。監査しやすい文書、同系統の 2 モデル、アンサンブル評価を含む。

<<<<<<< Updated upstream
**クイックリンク**
- 英語入口: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- 日本語入口: [isles2022_25d/README.md](isles2022_25d/README.md)
- 実験詳細: [isles2022_25d/README.md](isles2022_25d/README.md)
- 引用情報: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
=======
> 研究およびポートフォリオ提示を目的とする。**医療機器ではありません。** 医療データ本体は公開物に含めない。

**関連リンク**

- 英語版入口: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- 日本語版詳細: [isles2022_25d/README.md](isles2022_25d/README.md)
- 再現性点検表: [docs/reproducibility_checklist_ja.md](docs/reproducibility_checklist_ja.md)
- スナップショット案内: [docs/releases/isles2022-2p5d-v1.2-portfolio_ja.md](docs/releases/isles2022-2p5d-v1.2-portfolio_ja.md)
- GitHub About: [docs/github_about_ja.md](docs/github_about_ja.md)
- 引用: [CITATION.cff](CITATION.cff)
>>>>>>> Stashed changes
- ロードマップ: [ROADMAP.md](ROADMAP.md)
- 公開整備チェック表: [docs/audit/polish-isles2022-2p5d.md](docs/audit/polish-isles2022-2p5d.md)

## 本リポジトリで確認できること

<<<<<<< Updated upstream
- ISLES 2022 病変セグメンテーションの学習 → 評価 → アンサンブル評価ワークフロー
- 2 つのセグメンテーションモデルとして、近傍重視モデルと広域文脈モデルの設計差分を比較
- フル 3D より軽量な構成での病変セグメンテーション実験
- 外部レビュー向けに整理したポートフォリオ導線
- 実データなしで公開物の配線を確認できる簡易動作確認

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
=======
- 学習、評価、アンサンブル評価の流れ
- 近傍重視モデルと広域文脈モデルの比較
- しきい値スイープ、後処理スイープ、温度スケーリング、Dice 以外の指標
- 実データなしでの公開物の動作確認

### 指標の前提（公平な最終テスト条件）

- 検証で選んだ閾値: `0.75`
- `min_size=32`
- `prob_filter=0.90`

最終アンサンブルのテスト指標例: 平均 Dice 0.621、ボクセル適合率 0.856、ボクセル再現率 0.539。

## 処理構成

<p align="center"><img src="docs/assets/architecture_ja.png" alt="処理構成" width="100%" /></p>

<p align="center"><img src="docs/assets/repo_map_ja.png" alt="リポジトリ案内" width="100%" /></p>

<p align="center"><img src="docs/assets/results_snapshot_ja.png" alt="代表指標" width="100%" /></p>
>>>>>>> Stashed changes

### 代表指標

| 指標 | 値 | 意味 |
|---|---:|---|
<<<<<<< Updated upstream
| Local test mean Dice | 0.631 | 公開レシピの実用的な性能目安 |
| Validation mean Dice | 0.722 | 2 モデルを組み合わせたアンサンブルの検証性能 |
| 3D U-Net baseline test Dice | 0.514 | 局所比較の基準値 |
| Relative gain vs 3D baseline | +22.8% | 2.5D アンサンブルによる改善幅 |
=======
| ローカルテストの平均 Dice | 0.631 | 公開レシピの実用的な性能目安 |
| 検証の平均 Dice | 0.722 | 2 モデルを組み合わせたアンサンブルの検証性能 |
| ローカルテストの病変単位 F1 | 0.536 | Dice だけでは見えにくい病変単位の検出性能 |
| ローカルテストのボクセル再現率 | 0.539 | 最終アンサンブルの病変取りこぼしの補足 |
| 3D U-Net ベースラインのテスト Dice | 0.514 | 局所比較の基準値 |
| 3D ベースライン比の改善幅 | +22.8% | 2.5D アンサンブルによる改善幅 |
>>>>>>> Stashed changes

> 数値は同梱レシピと評価メモに基づく。医療データ本体は公開物に含めない。研究実装であり、**医療機器ではありません**。

<<<<<<< Updated upstream
## 最短の確認方法

### 1. 実データなしで配線確認
=======
## 医療データを用いず動作を確認する手順
>>>>>>> Stashed changes

```bash
python scripts/smoke_test.py --use_dummy_data
```

```bash
cd core/pipeline
python tools/make_manifest.py
```

実データによる学習・評価は [isles2022_25d/README.md](isles2022_25d/README.md) を参照する。

## 公開範囲

含むもの: ソース、設定、監査・評価文書、静的図表、リリースノート原稿。

<<<<<<< Updated upstream
含まれるもの:
- ソースコード
- 設定ファイル
- 監査 / 評価ドキュメント
- 静的図表と release note 原稿
=======
含めないもの: `Datasets/`、`runs/`、`results/`、`logs/`。
>>>>>>> Stashed changes

## スナップショット

<<<<<<< Updated upstream
## 固定スナップショット（ポートフォリオ用）

開発は継続中ですが、ポートフォリオ / 面接レビュー用の固定版は次のタグです。

✅ `isles2022-2p5d-v1.0-interview`
=======
ポートフォリオ用固定タグは `isles2022-2p5d-v1.2-portfolio` である。
>>>>>>> Stashed changes

## 引用

[CITATION.cff](CITATION.cff) に従う。

## コミットメッセージ

Conventional Commits（`fix:`、`feat:`、`docs:` 等）に統一する。
