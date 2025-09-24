# [PWS Cup 2025](https://www.iwsec.org/pws/2025/cup25.html) 用サンプルデータ・サンプルコード共有ディレクトリ

※ サンプルデータやサンプルコードは随時追加や更新されますので、更新日時を確認して最新版を利用するようにしてください。予備戦開始（8/20）後に追加または更新した場合はチーム代表者に連絡します。
## 重要な変更
- (9/19, 09:46) Ciの自己採点用スクリプトevaluation/eval_all.pyにBi.csvとCi.csvのフォーマットチェック機能を追加 [PR#38](https://github.com/pwscup/pwscup2025-scripts/pull/38)
- (9/19, 09:45) Ciの各列の値域確認用jsonファイルの参照先をdata/pre_columns_range.jsonからdata/columns_range.jsonに変更 [PR#39](https://github.com/pwscup/pwscup2025-scripts/pull/39)
- (9/18, 09:54) 初心者ガイドGUIDE_FOR_BEGINNERS.mdを追加 [PR#33](https://github.com/pwscup/pwscup2025-scripts/pull/33)
- (9/11, 08:25) attack/attack_Di.pyを改修。運営が意図しない特徴量を使うモデルDiに対してもattack_Di.pyとattack_example.pyが攻撃を強制的に実行 [PR#30](https://github.com/pwscup/pwscup2025-scripts/pull/30)
- (9/04, 21:06) util/check_csv.pyを修正。Ci.csvの値のチェック方法にバグがあったため修正。util/check_and_fix_csv.pyを追加。Ciの軽微なフォーマット違反を修正する。
- (8/29, 13:01) ↓の正規化方法が不適切で誤差が0になるべき入出力で誤差が0.5になる問題を修正 [PR#21](https://github.com/pwscup/pwscup2025-scripts/pull/21)
- (8/28, 15:16) evaluation/LR_asthma_diff.py のcoefに関する採点で正規化が行われていなかったため、正規化処理を追加 [PR#19](https://github.com/pwscup/pwscup2025-scripts/pull/19)

## ![PWS Cup 2025 の基本的な流れ](PWSCUP2025flow.pdf)

<img width="1050" height="567" alt="image" src="https://github.com/user-attachments/assets/859c85d5-c340-488e-bf68-3a58edc2e981" />

## Installation

```bash
pip install -r requirements.txt
```

