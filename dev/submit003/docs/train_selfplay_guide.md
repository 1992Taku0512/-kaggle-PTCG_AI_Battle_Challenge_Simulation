# AlphaZero Self-Play 学習実行手順書 (`train_selfplay_guide.md`)

本ドキュメントでは、`dev/submit003` に実装されている AlphaZero型 深層強化学習（DRL）思考エンジンの自己対戦（Self-Play）および GPU でのモデル重み更新スクリプト **`train_selfplay.py`** の実行手順・ハイパーパラメータ調整方法を解説します。

---

## 1. 概要

`train_selfplay.py` は、以下の処理を自動的にループ実行します。

1. **自己対戦 (Self-Play) & 対多様相手学習**: 自身のデッキを本番提出用の **`submit003/deck.csv` に固定**しつつ、対戦相手のデッキを `dev/deck_pool/`（10種類）からランダムに選出して対戦を周回。盤面状態（269次元テンソル）とMCTS探索結果のデータ（軌跡）を収集。
2. **GPU重み更新 (Backpropagation)**: 収集した対戦軌跡データを RTX 2070 SUPER（`cuda`）に送り、`AlphaZeroNet`（Policy + Value Dual-Head ネットワーク）の重みを更新。
3. **チェックポイント保存**: 更新された重みを `dev/submit003/model_weights.pt` に保存。
4. **LINE通知**: 学習完了・勝率・経過時間を LINE ボットへ自動送信。

---

## 2. 前提条件と環境確認

実行前に GPU (CUDA) が正常に利用可能か確認します。

```bash
# NVIDIA GPU 状態確認
nvidia-smi

# PyTorch CUDA 動作確認
uv run python test_gpu.py
```

---

## 3. 基本的な実行方法

リポジトリルートディレクトリ（`/mnt/d/work/git/kaggle_PTCG_AI_Battle`）で以下のコマンドを実行します。

```bash
uv run python dev/submit003/train_selfplay.py
```

---

## 4. ハイパーパラメータのカスタマイズ方法

`dev/submit003/train_selfplay.py` の `train_self_play()` 関数の引数を変更することで、学習規模や精度を調整できます。

### ⚙️ 主なハイパーパラメータ

| パラメータ | デフォルト値 | 推奨値（本格学習時） | 説明 |
| :--- | :---: | :---: | :--- |
| `num_episodes` | `5` | `50` 〜 `200` | 実行する自己対戦の試合数。多いほど多くの対戦経験を得られます。 |
| `num_simulations` | `20` | `40` 〜 `100` | 1手あたりのMCTS木探索回数。増やすと1手の思考精度が上がります。 |
| `batch_size` | `16` | `32` 〜 `128` | GPUに一度に投入する学習サンプル数。 |
| `epochs_per_episode` | `5` | `5` 〜 `10` | 1対戦完了ごとにGPUで重みを更新する周回数。 |

### 📝 コード書き換えによる変更例 (`train_selfplay.py` 最下部)

```python
if __name__ == "__main__":
    # 例：本格的な 50 試合の強化学習を実行する場合
    train_self_play(
        num_episodes=50,
        num_simulations=40,
        batch_size=64,
        epochs_per_episode=5
    )
```

---

## 5. 学習成果の確認・評価手順

### ① モデル重みファイルの確認
学習が成功すると、以下のパスに最新の重みが自動上書き保存されます。

- **`dev/submit003/model_weights.pt`**

### ② ローカル評価（`eval_local.py`）での勝率測定
学習した `submit003`（DRL AI）と `submit002`（ルールベース AI）を20試合対戦させて勝率を測定します。

```bash
uv run eval_local.py --agent1 dev/submit003 --agent2 dev/submit002 --num-games 20
```

---

## 6. トラブルシューティング

- **`CUDA out of memory` が出た場合**:
  - `batch_size` を `16` または `32` に下げてください。
  - `num_simulations` を `20`〜`30` に抑えてください。
- **学習ログ・LINE通知が届かない場合**:
  - ルート直下の `.env` ファイルに `LINE_ACCESS_TOKEN` と `LINE_USER_ID` が正しく設定されているか確認してください。
