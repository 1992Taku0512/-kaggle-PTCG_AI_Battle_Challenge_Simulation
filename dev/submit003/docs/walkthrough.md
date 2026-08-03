# dev/submit003 実装完了ドキュメント（AlphaZero 深層強化学習エージェント）

RTX 2070 SUPER（GPU）を活用した **AlphaZero型 深層強化学習（DRL）思考エンジン `dev/submit003`** の環境構築、モデル設計、評価ハーネスのプレイヤーインデックス抽出修正（`yourIndex` / `result`）、ベンチ構築ロジック修正、Policy Loss ＋ Value Loss 合成損失関数による 1,000エピソードGPU学習、および全対戦相手に対するベンチマーク評価が完了しました。

---

## 成果物一覧

- **[dev/deck_pool/](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/deck_pool/)**: 多様な属性・テーマ（草・炎・水・雷・超・闘・ロケット団・ex主軸・混合）で構成された全10種類の60枚デッキプール ＋ 解説書 (`README.md`)
- **[dev/submit003/deck.csv](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/deck.csv)**: 深層強化学習に最適化した60枚の単色・高火力exデッキ
- **[dev/submit003/state_encoder.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/state_encoder.py)**: 盤面状態（`Observation`）を269次元のPyTorchテンソルにエンコードするモジュール
- **[dev/submit003/model.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/model.py)**: Dual-Head（Policy ＋ Value）構造のAlphaZero用PyTorch GPUモデル
- **[dev/submit003/mcts.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/mcts.py)**: dict/dataclass対応 ＋ アグレッシブ行動priorボーナス搭載 MCTS探索エンジン
- **[dev/submit003/main.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/main.py)**: Kaggleコンテスト提出用のエージェントエントリーポイント
- **[dev/submit003/train_selfplay.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/train_selfplay.py)**: Replay Buffer ミニバッチSGD学習 ＋ 10種デッキプール対応の強化学習スクリプト
- **[dev/submit003/model_weights.pt](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/model_weights.pt)**: 1,000エピソードGPU学習済みの重みチェックポイント
- **[run_1000ep_benchmark.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/run_1000ep_benchmark.py)**: 1,000エピソード学習 ＋ 3ターゲット評価一括実行スクリプト
- **[dev/submit003/docs/train_selfplay_guide.md](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/docs/train_selfplay_guide.md)**: 学習実行手順書
- **[dev/submit003/docs/drl_algorithm_comparison.md](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/docs/drl_algorithm_comparison.md)**: DRL比較設計書
- **[dev/submit003/docs/walkthrough.md](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit003/docs/walkthrough.md)**: 成果・成果物まとめドキュメント

---

## 1,000エピソード GPU学習 ＆ ベンチマーク対戦評価結果

評価ハーネスとエージェント構築ロジックの修正完了後、1,000エピソードのGPU学習重みを用いて主要3エージェントそれぞれと20試合ずつ（計60試合）対戦評価を行いました。

- **学習規模**: 1,000 エピソード (Sliding Replay Buffer SGD)
- **GPU学習時間**: 113.6 秒 (約1.9分)
- **学習ファイル**: `dev/submit003/model_weights.pt`

### 📊 20試合対戦ベンチマーク勝率

| 対戦相手 | エージェントディレクトリ | 勝数 / 総試合数 | 勝率 (%) | 平均ターン数 | クラッシュ / エラー数 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kaggle Sample** | `data/sample_submission/sample_submission` | **6 / 20 勝** | **30.0%** | **49.6 ターン** | 0 |
| **submit001** | `dev/submit001` | **9 / 20 勝** | **45.0%** | **103.9 ターン** | 0 |
| **submit002** | `dev/submit002` | **12 / 20 勝** | **60.0%** | **100.8 ターン** | 0 |

---

## 🎯 7時間50,000エピソード特訓用 報酬（Reward Shaping）・ペナルティ設計一覧

単色高火力exデッキの長所（速攻攻撃・エネルギー育成）を最大限伸ばし、不要なパス行動を強力に抑制するために導入された設計一覧です：

| 区分 | アクション / 状態変化 | 報酬・罰則値 | 目的・効果 |
| :--- | :--- | :---: | :--- |
| 🟢 **終局** | **対戦勝利 (Match Victory)** | **`+5.0`** | 勝率向上を強力に誘引する重量級勝利報酬 |
| 🔴 **終局** | **対戦敗北 (Match Defeat)** | **`-5.0`** | 敗北回避の強いペナルティ |
| 🟢 **盤面** | **相手ポケモン気絶 (Prize Card Taken)** | **`+1.5` / 枚** | サイド獲得・敵撃破へのプラス評価 |
| 🔴 **盤面** | **自ポケモン気絶 (Prize Card Lost)** | **`-1.5` / 枚** | 自ポケモン気絶の防止 |
| 🟢 **戦術** | **ワザ使用・攻撃成功 (OptionType 8)** | **`+0.3`** | ダメージを与える積極攻撃への加点 |
| 🟢 **戦術** | **エネルギーアタッチ / たね出撃 / 進化** | **`+0.2`** | 攻撃準備・ベンチ展開（0体負け防止） |
| 🟢 **戦術** | **トレーナーズ効果プレイ (OptionType 2)** | **`+0.2`** | 手札リソースの有効活用 |
| 🔴 **反則手** | **無駄なパス (Unnecessary Turn End)** | **`-0.5`** | **攻撃やエネルギーアタッチが可能な場面でのPASS**に対する罰則 |

---

## LINE通知

- 1時間ごとの定期進捗レポート、および 50,000エピソード学習完了後の最終ベンチマーク対戦結果（全60試合）と提出用ZIP自動パッケージング完了を LINE へ送信いたします。
