# Public repo polish checklist — isles2022-2p5d-reproducible-pipeline

テンプレ由来: RSNA `docs/audit/PUBLIC_REPO_POLISH_CHECKLIST.md`  
監査日: 2026-11-11  
対象ブランチ: `main`（作業は main に統合）

## 1. 証拠と主張

| 項目 | 状態 | メモ |
|---|---|---|
| 数値の一次成果からの再計算 | 該当なし / 部分 | README 表と評価メモ依存 |
| 制約と数値の同一節 | 完了 | caveat + 指標表 |
| claims 検査スクリプト | 該当なし | 未移植 |

## 2. 公開衛生

| 項目 | 状態 | メモ |
|---|---|---|
| 機微パス検査 | 部分 | ignore 規則で除外。専用スクリプトなし |
| 相対リンク | 完了 | |
| 無視規則 | 完了 | |

## 3. 図

| 項目 | 状態 | メモ |
|---|---|---|
| PNG + `width="100%"` | 完了 | |
| `*_ja.png` | 完了 | |
| Mermaid 非依存 | 完了 | |
| raw ハッシュ照合 | 未 | push 後 |

## 4. 日本語

| 項目 | 状態 | メモ |
|---|---|---|
| である調 | 完了 | README_ja 書き換え |
| textlint 設定 | 完了 | `.textlintrc.json` |
| 口語時間売り禁止 | 完了 | |

## 5. 完了定義

| 項目 | 状態 | メモ |
|---|---|---|
| ローカル図再生成 | 完了 | |
| raw 照合 | 未 | |
| ブラウザ確認 | 未 | |

## 監査時ギャップ（整備前）

- です・ます調の README_ja
- SVG 中心の埋め込み
- 作業ブランチが `dev` に偏り、`main` と乖離しやすい
- textlint なし、medical-device caveat が弱い

## 適用コミット

（push 後追記）
