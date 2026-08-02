# dev/submit004 実装＆特訓完了ドキュメント（250,000エピソード超高強度 AlphaZero）

`dev/submit003` をベースに発展させ、**中間報酬（Reward Shaping）・エネルギー効率化ペナルティ・50,000 ＋ 追加200,000（累計250,000エピソード）GPU超高強度特訓** を完走した最先端AlphaZero強化学習エージェント **`dev/submit004`** の成果ドキュメントです。

---

## 成果物一覧 (`dev/submit004/`)

- **[dev/submit004/deck.csv](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/deck.csv)**: 単色・高火力exベース60枚構築デッキ
- **[dev/submit004/state_encoder.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/state_encoder.py)**: 269次元盤面エンコーダー
- **[dev/submit004/model.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/model.py)**: Dual-Head (Policy / Value) AlphaZero GPUネットワーク
- **[dev/submit004/mcts.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/mcts.py)**: Priorボーナス ＋ ディリクレノイズ MCTS探索エンジン (`num_simulations=150`)
- **[dev/submit004/main.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/main.py)**: Kaggle環境互換（`cg`同梱・`__file__`非依存）エントリーポイント
- **[dev/submit004/train_selfplay.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/train_selfplay.py)**: 250,000エピソード特訓スクリプト
- **[dev/submit004/model_weights.pt](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/model_weights.pt)**: 累計250,000エピソードGPU学習済み重み
- **[dev/submit004/submission.zip](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit004/submission.zip)**: Kaggle提出用完全パッケージ (7.46 MB, cgモジュール同梱版)

---

## 🏆 累計250,000エピソード 最終ベンチマーク成績（全60試合）

| 対戦相手 | エージェントディレクトリ | 勝数 / 総試合数 | 勝率 (%) | 平均ターン数 | クラッシュ / エラー数 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **submit002** (上位ルールベース) | `dev/submit002` | **13 / 20 勝** | **65.0%** 🏆 | **89.0 ターン** | 0 |
| **submit001** (標準ルールベース) | `dev/submit001` | **10 / 20 勝** | **50.0%** | **95.2 ターン** | 0 |
| **Kaggle Sample** (サンプル) | `data/sample_submission/sample_submission` | **8 / 20 勝** | **40.0%** | **45.6 ターン** | 0 |

- **GPU累計学習時間**: 約 6.8 時間 (計 250,000 試合 Self-Play)
- **探索数**: 150 回 / 手

---

## 🎯 適用された報酬（Reward Shaping）・ペナルティ設計表

| 区分 | アクション / 状態変化 | 報酬 / 罰則値 | 条件・判定タイミング | 目的・設計の理由 |
| :--- | :--- | :---: | :--- | :--- |
| 🟢 **終局** | **対戦勝利 (Match Victory)** | **`+5.0`** | ゲーム終了時に勝利 | 勝率向上を強力に誘導する最優先報酬。 |
| 🔴 **終局** | **対戦敗北 (Match Defeat)** | **`-5.0`** | ゲーム終了時に敗北 | 敗北回避のための強力なペナルティ。 |
| 🟢 **盤面** | **相手ポケモン気絶 (Enemy KO)** | **`+1.5` / 枚** | 相手サイドが減った瞬間 | サイド獲得・相手主力撃破への積極評価。 |
| 🔴 **盤面** | **自ポケモン気絶 (Own KO)** | **`-1.5` / 枚** | 自分のサイドが減った瞬間 | 自ポケモン気絶の防止。 |
| 🟢 **戦術** | **ワザ攻撃成功 (ATTACK)** | **`+0.3`** | `OptionType.ATTACK` (13) | ダメージを与える攻撃行動への評価。 |
| 🟢 **戦術** | **適正エネルギーアタッチ** | **`+0.3`** | 技コスト枠へのエネルギー手貼り | **技発動準備の効率化**。 |
| 🔴 **非効率** | **上限超えエネルギーアタッチ** | **`-0.2`** | **コスト満了ポケモンへの追加手貼り** | **エネ無駄貼りの抑制**。手札消耗防止。 |
| 🔴 **非効率** | **属性ミスマッチエネアタッチ** | **`-0.3`** | **技タイプ不一致エネ手貼り** | **無効エネ貼りの抑制**。 |
| 🟢 **戦術** | **たね出撃 / 進化** | **`+0.2`** | たね出撃 / 進化実行 | アタック準備・ベンチ展開（0体負け防止）。 |
| 🟢 **戦術** | **トレーナーズプレイ (PLAY)** | **`+0.2`** | `OptionType.PLAY` (7) | 手札リソースの有効活用。 |
| 🔴 **反則手** | **無駄なパス (Unnecessary PASS)** | **`-0.5`** | **攻撃や有効手貼りが可能なのにPASS** | 目先の迷いによる不必要なターンパスを抑制。 |
