# dev/submit005 実装＆学習設計ドキュメント（4段階適応型カリキュラム AlphaZero）

`dev/submit004` の知見（対不効率手ペナルティ・150回MCTS探索・`cg`同梱互換性）を継承し、**新規初期化モデルによる4段階適応型カリキュラム学習（4-Phase Adaptive Curriculum DRL）** を導入した最先端強化学習エージェント **`dev/submit005`** の概要ドキュメントです。

---

## 成果物一覧 (`dev/submit005/`)

- **[dev/submit005/deck.csv](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/deck.csv)**: 単色・高火力exベース60枚構築デッキ
- **[dev/submit005/state_encoder.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/state_encoder.py)**: 269次元盤面エンコーダー
- **[dev/submit005/model.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/model.py)**: Dual-Head (Policy / Value) AlphaZero GPUネットワーク
- **[dev/submit005/mcts.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/mcts.py)**: Priorボーナス ＋ ディリクレノイズ MCTS探索エンジン (`num_simulations=150`)
- **[dev/submit005/main.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/main.py)**: Kaggle環境互換（`cg`同梱・`__file__`非依存）エントリーポイント
- **[dev/submit005/train_curriculum.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/train_curriculum.py)**: 4段階適応型カリキュラム学習 ＋ 1時間LINE通知 ＋ 各フェーズ自動対戦評価スクリプト
- **[dev/submit005/model_weights.pt](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/model_weights.pt)**: 新規初期化 ➡ カリキュラム学習済みモデル重み
- **[dev/submit005/submission.zip](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit005/submission.zip)**: Kaggle提出用完全パッケージ (cgモジュール同梱版)

---

## 🎯 4段階適応型カリキュラムの勝率昇格設計

| フェーズ | モデル | 対戦相手デッキ | 対戦相手AI方策 | フェーズ自動昇格 / 完成条件 | 主な学習目標 |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Phase 1** | **新規初期化** | **自分のデッキのみ** (ミラー戦) | Self-Play | **直近200試合 勝率 ≥ 85.0%**<br>(平均ターン数 ≤ 25T) | **基礎の確立**。自分のデッキの最速手貼り・最速攻撃・ノーミス盤面展開の暗記。 |
| **Phase 2** | 引き継ぎ | **主要3属性** (炎・水・雷) | Self-Play + **submit001** | **対 submit001 勝率 ≥ 75.0%** | **属性対応の基礎**。タイプ相性に応じた育成優先度の切り替えを学習。 |
| **Phase 3** | 引き継ぎ | **主要6属性** | Self-Play + **submit002** | **対 submit002 勝率 ≥ 68.0%** | **上位AI超克**。強豪ルールベースAI (`submit002`) に安定勝ち越し。 |
| **Phase 4** | 引き継ぎ | **全10種デッキプール** | Self-Play + submit001 + submit002 + **submit004重み** | **全対戦相手 総合勝率 ≥ 60.0%** | **総力戦・完成**。過去最高モデル(`submit004`)と全AIを克服した汎用性の完成。 |

※ 各フェーズの安全上限は **最大 1,000,000 エピソード** とし、勝率条件を満たすまでじっくり学習を継続可能です。

---

## 📊 適用する報酬（Reward Shaping）・ペナルティ設計表

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

---

## ⏰ LINE報告 ＆ 各フェーズ完了自動ベンチマーク
- 1時間ごとに **現在のPhase・累計エピソード・直近勝率・平均ターン数・Loss** をLINEへ自動送信。
- フェーズ1、フェーズ2、フェーズ3、フェーズ4の昇格・達成毎に、`Kaggle Sample`, `submit001`, `submit002` との計60試合対戦ベンチマークおよび `submission.zip` パッケージングを自動実行。
