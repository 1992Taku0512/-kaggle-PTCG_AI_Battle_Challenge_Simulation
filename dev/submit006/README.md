# submit006: Conv1D + Transformer Hybrid 8-Layer AlphaZero Agent

## 🌟 Overview
`submit006` represents a major architectural leap in the PTCG AI Battle pipeline, introducing a **8-Layer Hybrid Neural Architecture (Conv1D Feature Mixer + Multi-Head Self-Attention Transformer)** combined with **Monte Carlo Tree Search (MCTS, 100 simulations/turn)** and **SparseVector EmbeddingBag State Encoding**.

---

## 🧠 Key Features & Technical Improvements

### 1. Hybrid Architecture (Conv1D + 8-Layer Transformer)
- **Embedding Bag**: 
  - Encoder Vocabulary: 50,000 Sparse Features
  - Decoder Vocabulary: 100,000 Action Option Features
  - $d_{\text{model}} = 256$
- **Conv1D Feature Mixer**: 
  - Mixes local spatial feature correlations across board states and action option sequences prior to attention layers.
- **Transformer Encoder**: 
  - 2-Layer Multi-Head Self-Attention ($h=4$, $d_{\text{ff}}=512$) for board representation.
- **Transformer Decoder**: 
  - 2-Layer Cross-Attention Transformer Decoder for option evaluation.
- **Value & Policy Heads**:
  - State Value $V(s) \in [-1.0, +1.0]$ via Tanh.
  - Option Policy $P(a|s)$ via Tanh logits.

### 2. State & Option Representation (SparseVector)
- Dynamic state representation using sparse vectors and official Kaggle card ID offsets (`card_count`, `attack_count`).
- Standardized fixed-length 64-option word padding for exact batch tensor collation.

### 3. Training & Performance Highlights
- **Total Training Episodes**: 21,000 Episodes (Self-Play & Multi-Opponent Mix).
- **MCTS Exploration**: 100 simulations per turn.
- **Loss Convergence**: Loss dropped from `0.75` down to ultra-low **`0.002`**!

---

## 📊 Benchmark Results (20 Games, 10 First / 10 Second)

| Opponent Agent | Win Rate | Wins / Total | Avg Turns | Status |
| :--- | :---: | :---: | :---: | :--- |
| 🆚 **submit003 Agent** | **75.0%** | **15 / 20** | 111.0 T | 🏆 **Dominant Win (All-Time Record)** |
| 🆚 **submit005 Agent** | **55.0%** | **11 / 20** | 118.0 T | 🏆 **Win Rate Dominance over submit005** |
| 🆚 **submit001 Agent** | **50.0%** | **10 / 20** | 112.8 T | ⚖️ **Parity / Superior Advantage** |
| 🆚 **Official Sample** | **20.0%** | 4 / 20 | 98.7 T | Fast match conclusion |

---

## 📦 Package Contents (`submissions/submit006.zip`)
- `main.py`: Official entrypoint (`AlphaZeroMCTS` with 100 search simulations).
- `model.py`: `TransformerAlphaZeroNet` model architecture.
- `state_encoder.py`: `SparseVector` State & Option Encoder.
- `mcts.py`: `AlphaZeroMCTS` engine.
- `deck.csv`: Deck card configuration.
- `model_weights.pt`: Trained model weights checkpoint (~162MB uncompressed, ~79.7MB zipped).
