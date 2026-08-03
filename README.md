# kaggle_PTCG_AI_Battle

Kaggle Pokémon Trading Card Game (PTCG) AI Battle Challenge 用のシミュレーションおよびエージェント開発リポジトリです。

---

## 📁 ディレクトリ構成

- **`dev/submit001/`**: 初回作成したルールベースエージェント
  - `main.py`: 攻撃・たねポケモン・エネルギー配置・進化を優先するルールベース思考エンジン
  - `deck.csv`: 対戦用デッキ定義（60枚）
  - `docs/walkthrough.md`: 成果・実装内容ドキュメント
- **`data/sample_submission/sample_submission/`**: 公式サンプルエージェント（ランダム思考）および C++ ゲームエンジン（`cg` ライブラリ）
- **`eval_local.py`**: ローカル環境用対戦・評価評価スクリプト

---

## 🚀 実行方法・使い方

### 1. 単発対戦シミュレーションの実行

サンプルエージェント同士の対戦テスト：

```bash
uv run run_match.py
```

### 2. ローカル評価スクリプト (`eval_local.py`)

指定した2つのエージェント間で複数回の自動対戦を行い、勝率やエラー発生率、平均ターン数などの統計情報を集計します。先攻・後攻の有利不利を抑えるため、試合数の半分ずつプレイヤーの先攻・後攻を交互に入れ替えます。

```bash
# 例: dev/submit001 と 公式サンプルエージェント で 20 試合対戦評価
uv run eval_local.py --agent1 dev/submit001 --agent2 data/sample_submission/sample_submission --num-games 20
```

#### 引数オプション:
- `--agent1`: エージェント1のディレクトリパス (デフォルト: `dev/submit001`)
- `--agent2`: エージェント2のディレクトリパス (デフォルト: `data/sample_submission/sample_submission`)
- `--num-games`: 対戦回数 (デフォルト: `20`)
- `--max-turns`: 1試合あたりの上限ターン数 (デフォルト: `1000`)

---

## 🧠 汎用強化学習フレームワーク (`PTCGTrainer`)

`dev/common/` に構築された、任意のニューラルネットワーク構造や特徴量エンコーダーに対応する汎用強化学習（DRL）パイプラインです。

### 🌟 主な特徴

1. **モデル構造に依存しない設計 (Model-Agnostic)**:
   - 任意の PyTorch モデル（`nn.Module`）を引数として渡すことで、モデル構造変更時にも学習コードの書き換えが不要。
2. **ドメインランダム化 (デッキ ✕ 対戦相手の同時ランダム選出)**:
   - **デッキランダム**: 候補デッキプール (`dev/deck_pool/`) から毎対局 P1/P2 のデッキを動的抽出。
   - **対戦相手ランダム**: 自己対局 (`self_play`)、ランダムエージェント (`random`)、過去モデル (`past_checkpoint`) を任意の確率比率で選出。
3. **新規作成 ✕ 追加学習モード**:
   - `resume_checkpoint_path=None` で新規重み初期化。
   - チェックポイントパスを指定すると重み・Optimizer・エポック数を自動復元して再開（ファインチューニング）。
4. **自動評価 & LINE通知**:
   - 定期的に `eval_local.py` サブプロセスを実行してベンチマーク勝率を算出し、最高モデルを自動保存＆LINE通知。

### 🏃 実行方法

#### サンプル実行 (動作確認)
```bash
uv run python dev/common/example_train.py
```

#### スクリプトでの使用例
```python
from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit005.model import AlphaZeroNet
from dev.submit005.state_encoder import StateEncoder
from dev.submit005.mcts import AlphaZeroMCTS

# 1. 学習・ランダム化の設定
config = TrainerConfig(
    experiment_name="submit005_fresh_2000ep",
    resume_checkpoint_path=None,  # 新規作成 (追加学習の場合は ".pt" パスを指定)
    opponent_types=["self_play", "random", "past_checkpoint"],
    opponent_weights=[0.7, 0.15, 0.15],
    p1_deck_mode="random",
    p2_deck_mode="random",
    num_episodes=2000,
    eval_every=500,
    use_line_notify=True
)

# 2. モデルとエンコーダーの作成
model = AlphaZeroNet()
encoder = StateEncoder()

# 3. トレーナーの起動 (モデルを引数として注入)
trainer = PTCGTrainer(
    config=config,
    model=model,
    encoder=encoder,
    mcts_cls=AlphaZeroMCTS
)

# 4. 学習の開始
trainer.train(agent_save_dir="dev/submit005")
```

