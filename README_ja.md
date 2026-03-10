# isles2022-25d-pipeline

**言語:** 日本語 | [English](README.md)

ISLES 2022 虚血性脳卒中病変セグメンテーション — **2.5D ConvNeXt アンサンブル**による test mean Dice **0.631** (3D U-Net 単体比 +22.8%)。

**クイックリンク**
- 日本語詳細: [isles2022_25d/README.md](isles2022_25d/README.md)
- 英語版詳細: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- 引用情報: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)

## このリポジトリでできること

- ISLES 2022 脳梗塞病変セグメンテーションの 2.5D ConvNeXt パイプライン
- 2 モデル構成 (v2: 5-slice / v3: 7-slice dilated) と確率マップアンサンブル
- 小病変の見逃し軽減のための Tversky loss + EMA
- 学習 → 単一モデル評価 → アンサンブル評価のエンドツーエンドワークフロー
- 外部レビュー向けに整理したポートフォリオ導線
- 実データなしで公開物の配線を確認できる簡易動作確認

## 想定している読者

- 医療AIセグメンテーション実装を確認したい採用担当
- 軽量な 2.5D 畳み込みセグメンテーションを探している ML エンジニア
- ISLES 2022 の改善ベースラインを探している研究者

## 3分で分かる概要

![2.5D ConvNeXt アーキテクチャ](docs/assets/architecture.svg)

![リポジトリ構成](docs/assets/repo_map.svg)

![指標サマリー](docs/assets/results_snapshot.svg)

### 成果サマリ

| モデル | val mean Dice | test mean Dice |
|--------|:---:|:---:|
| 3D U-Net ベースライン | 0.652 | 0.514 |
| ConvNeXt 2.5D v2 (5-slice) | 0.704 | ~0.58 |
| ConvNeXt 2.5D v3 (7-slice dilated) | 0.690 | 0.579 |
| **v2 + v3 アンサンブル** | **0.722** | **0.631** |

3D U-Net 単体比: **test +0.117 (+22.8%)**

> 数値は ISLES 2022 分割での局所評価に基づきます。医療データ本体は公開物に含めていません。

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

- 日本語詳細ガイド: [isles2022_25d/README.md](isles2022_25d/README.md)
- 英語版詳細ガイド: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)

## 含まれるものと含まれないもの

含まれるもの:
- ソースコード (モデル・データセット・学習・評価)
- 設定ファイル (v2 / v3 / vanilla 2.5D UNet)
- ドキュメントとリリースノート原稿
- 静的図表

含まれないもの:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## 関連リポジトリ

- **3D U-Net ベースライン**: [isles2022-3d-reproducible-pipeline](https://github.com/yskfuji/isles2022-3d-reproducible-pipeline)

## 引用

[CITATION.cff](CITATION.cff) を参照してください。

## コミットメッセージの規約

今後の変更は Conventional Commits (`type: summary`) で揃えます。

- `fix: eval スクリプトの閾値デフォルト値修正`
- `feat: TTA サポート追加`
- `refactor: データセットローダー整理`
- `docs: v2 vs v3 config 差異の明確化`
