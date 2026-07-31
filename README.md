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
