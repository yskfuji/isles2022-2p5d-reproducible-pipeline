# isles2022-2p5d-reproducible-pipeline

**言語:** 日本語 | [英語版](README.md)

ISLES 2022 向けの、**再現可能な 2.5D 脳梗塞病変セグメンテーションパイプライン**です。監査しやすいドキュメント、同系統の 2 モデル、アンサンブル評価を含みます。

**クイックリンク**
- 英語版: [isles2022_25d/README_en.md](isles2022_25d/README_en.md)
- 日本語版: [isles2022_25d/README.md](isles2022_25d/README.md)
- 実験詳細: [isles2022_25d/README.md](isles2022_25d/README.md)
- 再現性チェックリスト: [docs/reproducibility_checklist_ja.md](docs/reproducibility_checklist_ja.md)
- レビューの見始め方: [英語版スナップショット案内](docs/releases/isles2022-2p5d-v1.2-portfolio.md) | [日本語版スナップショット案内](docs/releases/isles2022-2p5d-v1.2-portfolio_ja.md)
- GitHub About 欄の説明文: [英語版](docs/github_about.md) | [日本語版](docs/github_about_ja.md)
- 引用情報: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [英語版](docs/releases/isles2022-2p5d-v1.2-portfolio.md) | [日本語版](docs/releases/isles2022-2p5d-v1.2-portfolio_ja.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)

## このリポジトリでできること

- ISLES 2022 病変セグメンテーションの学習 → 評価 → アンサンブル評価ワークフロー
- 2 つのセグメンテーションモデルとして、近傍重視モデルと広域文脈モデルの設計上の違いを比較
- フル 3D より軽量な構成での病変セグメンテーション実験
- しきい値スイープ、後処理スイープ、確率マップの再利用、温度スケーリング、Dice 以外の追加指標を含む評価ツール群
- 外部レビュー向けに整理したポートフォリオ用ドキュメント
- 実データなしで公開物が正しく動くかを確かめる簡易動作確認

### Dice 以外も含む指標スナップショット

公平な最終テスト条件:
- 検証で選んだ閾値: `0.75`
- `min_size=32`
- `prob_filter=0.90`

最終アンサンブルのテスト指標:

| 指標 | 値 |
|---|---:|
| 平均 Dice | 0.621 |
| ボクセル適合率 | 0.856 |
| ボクセル再現率 | 0.539 |
| 病変単位の micro F1 | 0.536 |
| ASSD | 6.89 mm |
| HD95 | 21.40 mm |
| 平均絶対体積誤差 | 15.95 mL |

比較サマリー:

| データ分割 | モデル | Dice | 適合率 | 再現率 | 病変 F1 |
|---|---|---:|---:|---:|---:|
| 検証 | 近傍重視モデル | 0.704 | 0.859 | 0.813 | 0.513 |
| 検証 | 広域文脈モデル | 0.713 | 0.830 | 0.827 | 0.543 |
| 検証 | アンサンブル | 0.722 | 0.803 | 0.865 | 0.615 |
| テスト | 近傍重視モデル | 0.588 | 0.874 | 0.449 | 0.416 |
| テスト | 広域文脈モデル | 0.624 | 0.892 | 0.485 | 0.471 |
| テスト | アンサンブル（公平な最終値） | 0.621 | 0.856 | 0.539 | 0.536 |

詳細な検証 / テスト比較表: [artifacts/eval_runs/metrics_comparison_20260311.md](artifacts/eval_runs/metrics_comparison_20260311.md)

## 内部ファイル名の対応表

設定ファイル名や学習済み重みの例では、過去の内部実験名が残っています。両者とも基本構成は同じで、ConvNeXt エンコーダと U-Net 風デコーダを使います。

| ファイル中の内部名 | 公開文書での意味 |
|---|---|
| `convnext_v2_5slice_1mm` | 近傍重視モデル（5 スライス） |
| `convnext_v3_7slice_dilated_1mm` | 広域文脈モデル（7 スライス・拡張間隔） |

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
| ローカルテストの平均 Dice | 0.631 | 公開レシピの実用的な性能目安 |
| 検証の平均 Dice | 0.722 | 2 モデルを組み合わせたアンサンブルの検証性能 |
| ローカルテストの病変単位 F1 | 0.536 | Dice だけでは見えにくい病変単位の検出性能 |
| ローカルテストのボクセル再現率 | 0.539 | 最終アンサンブルの病変取りこぼしの少なさを補足 |
| 3D U-Net ベースラインのテスト Dice | 0.514 | 局所比較の基準値 |
| 3D ベースライン比の改善幅 | +22.8% | 2.5D アンサンブルによる改善幅 |

> 数値は同梱レシピと評価メモに基づきます。医療データ本体は公開物に含めていません。

## 最短の確認方法

### 1. 実データなしで動作確認

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
- 静的図表とリリースノート原稿

含まれないもの:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## 現行のポートフォリオ用スナップショット

開発は継続中ですが、現行のポートフォリオ用固定スナップショットは次のタグです。

✅ `isles2022-2p5d-v1.2-portfolio`

## 引用

[CITATION.cff](CITATION.cff) を参照してください。

## コミットメッセージの規約

今後の変更は Conventional Commits（`type: summary`）で揃えます。

- `fix: eval スクリプトの閾値デフォルト値修正`
- `feat: add ensemble calibration notes`
- `refactor: slice sampler validation`
- `docs: clarify 5-slice vs 7-slice evaluation protocol`
