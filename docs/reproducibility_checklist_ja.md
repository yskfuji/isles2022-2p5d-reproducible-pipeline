# 再現性チェックリスト

このページは、公開版 ISLES 2022 2.5D リポジトリを外部レビューする際の簡易チェックリストです。

## 1. パッケージの完全性

- README とリリースノートで参照している安定スナップショットのタグを確認する。
- リポジトリに機微な医療データや private な学習 artifact が含まれていないことを確認する。
- 必要に応じて、`core/pipeline` で `python scripts/smoke_test.py --use_dummy_data` または `python tools/make_manifest.py` を実行し、新しいマニフェストを生成する。

## 2. ドキュメントの整合性

- 入口となるページとして `README.md` または `README_ja.md` を読む。
- タスク説明として `isles2022_25d/README_en.md` または `isles2022_25d/README.md` を読む。
- リリースノート原稿として `docs/releases/isles2022-2p5d-v1.2-portfolio_ja.md` を確認する。
- 主要指標、アンサンブル構成、モデル名が、これらのファイル間で矛盾していないことを確認する。

## 3. コード経路の妥当性

- 学習と評価のエントリポイントが存在することを確認する。
  - `core/pipeline/src/training/train_isles_25d_convnext_fpn.py`
  - `core/pipeline/src/evaluation/evaluate_isles_25d.py`
  - `core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py`
- しきい値スイープ、後処理スイープ、確率マップ再利用、温度スケーリングが公開ツールとして含まれていることを確認する。

## 4. スモークテスト

- `python scripts/smoke_test.py --use_dummy_data` を実行する。
- 実データなしでコマンドが完了することを確認する。
- 出力された summary が、想定どおりの公開ファイルとエントリポイントを指していることを確認する。

## 5. 評価レシピとしての見やすさ

- README に Dice 以外の指標も明示されていることを確認する。
- 最終アンサンブルのレシピが、しきい値や後処理設定を含めて説明されていることを確認する。
- 内部実験名と公開向けの名称の対応が分かることを確認する。

## 6. モデル登録の検証

- リポジトリ直下では `python core/pipeline/tools/verify_registration.py --run-dir <REPRESENTATIVE_RUN_DIR> --model-name isles-25d-convnext --checkpoint best.pt --promotion-rule "val_dice>=0.72" --registered-model-name isles-25d-convnext-verify`、`core/pipeline` では `python tools/verify_registration.py ...` を実行する。
- `artifacts/verification/registered_models/.../registration.json` が生成されることを確認する。
- 標準出力の JSON 要約に、想定どおりの `promotion_status` と alias 結果が含まれることを確認する。

## 7. レビュワー目線の合格条件

- 2 モデル構成とアンサンブルの役割を 3 分以内に把握できる。
- train → evaluate → ensemble の流れを、非公開スクリプトに頼らず追える。
- 保護データがなくても、リポジトリの配線と再現性の考え方を確認できる。

## 8. 既知の制約

- このチェックリストが検証するのは、公開物としての再現性導線であり、実データでの end-to-end 学習そのものではない。
- 完全な指標再現には、別途準備した ISLES 2022 データと checkpoint が必要になる。