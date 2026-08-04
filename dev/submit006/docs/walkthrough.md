# dev/submit006 (Conv1D ✕ Transformer Hybrid 8層 DRL エージェント) 実装構造＆仕様ドキュメント

本ドキュメントでは、Kaggle 公式サンプルノートブック (`samples/Reinforcement_Learning_and_MCTS_sample_code/`) の洗練された構造を取り入れ構築した **`dev/submit006`** の設計思想・ニューラルネットワーク層構造・公式 Search API 連携仕様・使い方について詳細に解説します。

---

## 📌 アーキテクチャの概要と先進性

`dev/submit006` は、従来の固定長 MLP（ResBlock）アーキテクチャから脱却し、ポケモンカードゲーム（PTCG）のゲーム的特徴（カード組み合わせの多様性・動的選択肢・非完全情報）に完全に適合した **「Conv1D ✕ Transformer Hybrid 8層構造」** を採用しています。

### 🌟 3つの革新的機能
1. **EmbeddingBag ✕ SparseVector による動的カード埋め込み**:
   - 全 22,000 語彙（Encoder）および 50,000 語彙（Decoder）の埋め込み空間を構築。
   - 数千種類のカードIDやワザIDを疎行列（`SparseVector`）として表現し、次元爆発を起こさずに高度なカード文脈を獲得。
2. **Conv1D 特徴抽出層 (Pointwise Convolution)**:
   - 埋め込みチャネル（256次元）に対して `Conv1D(kernel_size=1)` を適用し、局所的なカード特徴相互作用（Feature Mixing）を実行。
3. **Transformer Cross-Attention による動的アクション選択肢アテンション**:
   - `MultiheadAttention` (4ヘッド) を使用し、現在提示されている行動選択肢（`SelectContext`）と「盤面全体の文脈」との間で Cross-Attention を計算。

---

## 🎨 ネットワーク構造の Mermaid フローチャート

```mermaid
graph TD
    subgraph Inputs ["1. 入力データ (Sparse Vector)"]
        EncInput["Encoder Input<br>盤面状態 (カード, 属性, リソース)<br>(Batch, 24, SparseIndices)"]
        DecInput["Decoder Input<br>行動選択肢 (SelectContext)<br>(Batch, Num_Options, SparseIndices)"]
    end

    subgraph Layer1 ["2. EmbeddingBag 層 (第1層 - 埋め込み)"]
        EncBag["encoder_bag: EmbeddingBag(22000, d_model=256)<br>→ (Batch, 24, 256)"]
        DecBag["decoder_bag: EmbeddingBag(50000, d_model=256)<br>→ (Batch, Num_Options, 256)"]
    end

    subgraph Layer2 ["3. Conv1D 特徴融合層 (第2層 - 局所特徴抽出)"]
        EncConv["Conv1D(256 -> 256, kernel=1) + LayerNorm + ReLU<br>→ チャネル間相互作用特徴"]
        DecConv["Conv1D(256 -> 256, kernel=1) + LayerNorm + ReLU<br>→ 行動チャネル間相互作用特徴"]
    end

    subgraph Layer3_4 ["4. Transformer Encoder (第3〜4層 - 2ブロック)"]
        TransEnc["TransformerEncoder (d_model=256, nhead=4, FFN=512, 2層)<br>→ 深層盤面文脈表現 encoder_out"]
    end

    subgraph Layer5_6 ["5. Decoder Cross-Attention (第5〜6層 - 2ブロック)"]
        CrossAttn["DecoderLayer (MultiheadAttention, 2層)<br>Query: 行動特徴 (DecConv)<br>Key/Value: 盤面文脈 (encoder_out)<br>→ アテンション特徴量"]
    end

    subgraph Layer7_8 ["6. Dual-Head 出力層 (第7〜8層 - 線形出力)"]
        ValueHead["Value Head: Linear(256 -> 1) + Tanh<br>→ 盤面勝率予測 V(s) ∈ [-1.0, +1.0]"]
        PolicyHead["Policy Head: Linear(256 -> 1) + Tanh<br>→ 行動選択肢優先度 P(a|s)"]
    end

    EncInput --> EncBag --> EncConv --> TransEnc
    DecInput --> DecBag --> DecConv --> CrossAttn
    TransEnc -. "encoder_out (Key/Value)" .-> CrossAttn

    TransEnc --> ValueHead
    CrossAttn --> PolicyHead
```

---

## ⚡ 公式 Search API ✕ MCTS 連携シーケンス

```mermaid
sequenceDiagram
    autonumber
    participant Agent as MCTS Agent (mcts.py)
    participant API as 公式 Search API (cg.api)
    participant NN as Transformer NN (model.py)
    participant Cpp as C++ シミュレーションエンジン

    Agent->>API: search_begin(your_deck, opp_deck)
    API->>Cpp: 非公開カードの決定論的補完 (Determinization)
    API-->>Agent: 仮想 SearchState ハンドルを取得

    loop N回 (例: 15〜40回) の MCTS 探索ループ
        Agent->>Agent: PUCT 式に基づいてツリー上の選択肢 a_t を決定
        Agent->>API: search_step(search_state, [action_idx])
        API->>Cpp: 仮想盤面上で行動 a_t を実行
        Cpp-->>API: 次の盤面状態 obs_next ＋ 行動選択肢
        API-->>Agent: obs_next を返却

        Agent->>NN: obs_next を入力して推論 (GPU)
        NN-->>Agent: 評価値 V(s) ＋ 各行動優先度 P(a|s)
        Agent->>Agent: MCTS ノードの訪問回数 N と累積価値 Q を逆伝播更新
    end

    Agent->>API: search_end(search_state)
    API->>Cpp: C++ 仮想シミュレータメモリの解放
    Agent->>Agent: 訪問回数 N が最大の「最良アクション」を決定
```

---

## 📁 成果物・ファイル一覧

| ファイル | パス | 役割 |
| :--- | :--- | :--- |
| **`state_encoder.py`** | [dev/submit006/state_encoder.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit006/state_encoder.py) | `SparseVector` エンコーダー（盤面・行動選択肢の数値化） |
| **`model.py`** | [dev/submit006/model.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit006/model.py) | `Conv1D` ＋ `TransformerEncoder` (2層) ＋ `DecoderLayer` (2層) 8層モデル |
| **`mcts.py`** | [dev/submit006/mcts.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit006/mcts.py) | `search_begin` / `search_step` / `search_end` 連携 MCTS 検索エンジン |
| **`main.py`** | [dev/submit006/main.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit006/main.py) | Kaggle 提出用メインエントリーポイント (`agent`) |
| **`deck.csv`** | [dev/submit006/deck.csv](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit006/deck.csv) | 60枚の対戦用デッキレシピ |
| **`train.py`** | [dev/submit006/train.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit006/train.py) | `PTCGTrainer` による段階的カリキュラム学習実行スクリプト |

---

## 🏃 実行手順・学習方法

### 1. 段階的（カリキュラム）学習の開始
`PTCGTrainer` を呼び出し、パラメータ指定の3段階カリキュラム（ミラーマッチ $\rightarrow$ 3主要属性 $\rightarrow$ 全10種デッキ ✕ 過去モデル）で学習を実行します：

```bash
uv run python dev/submit006/train.py
```

### 2. ローカル環境での対戦評価 (`eval_local.py`)
`submit006` と過去モデルまたはサンプルエージェントとの対戦評価：

```bash
uv run eval_local.py --agent1 dev/submit006 --agent2 data/sample_submission/sample_submission --num-games 20
```
