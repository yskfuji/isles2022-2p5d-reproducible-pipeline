# 公開整備記録 — isles2022-2p5d-reproducible-pipeline

手順の定義: [rsna-ich-reproducible-pipeline 公開整備プレイブック](https://github.com/yskfuji/rsna-ich-reproducible-pipeline/blob/main/docs/audit/PUBLIC_REPO_POLISH_PLAYBOOK.md)  
記録日: 2026-11-11  
参照ブランチ: `main`  
参照コミット: `97d1b0b`（整備本体は `292b31f`）

## 1. 証拠と主張

| 項目 | 判定 | メモ |
|---|---|---|
| 数値の一次成果からの再計算 | 該当なし / 部分 | README 表と評価メモに依存する |
| 制約と数値の同一節 | 合格 | caveat と指標表を併記 |
| claims 検査スクリプト | 該当なし | 未導入 |

## 2. 公開衛生

| 項目 | 判定 | メモ |
|---|---|---|
| 機微パス検査 | 部分 | ignore 規則で除外。専用スクリプトは未導入 |
| 相対リンク | 合格 | |
| 無視規則 | 合格 | |

## 3. 図

| 項目 | 判定 | メモ |
|---|---|---|
| PNG + `width="100%"` | 合格 | architecture / repo_map / results_snapshot |
| `*_ja.png` | 合格 | README_ja から参照 |
| Mermaid 非依存 | 合格 | 静的図 |
| raw ハッシュ照合 | 合格 | `architecture_ja.png` の SHA-256 が raw と一致 |

## 4. 日本語

| 項目 | 判定 | メモ |
|---|---|---|
| である調 | 合格 | README_ja をである調へ統一 |
| textlint 設定 | 合格 | `.textlintrc.json` |
| 口語時間売り禁止 | 合格 | |

## 5. 完了定義

| 項目 | 判定 | メモ |
|---|---|---|
| ローカル図再生成 | 合格 | |
| raw 照合 | 合格 | |
| ブラウザ確認 | 未確認 | 閲覧者側でハードリロードを推奨 |

## 公開面での到達点

- `main` 上の日英入口で PNG 埋め込みと medical-device caveat を揃える
- 内部実験名と公開向け名称の対応表を維持する
- 公開整備の判定を本記録に残す
