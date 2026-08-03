import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Map known card IDs in submit003 deck and general game cards to dense indices
CARD_VOCAB = [
    0,     # Unknown / Padding
    1002,  # ザングースex
    1006,  # メガタブンネex
    1010,  # ジジーロン
    1205,  # シアノ
    1182,  # ボスの指令
    1227,  # リーリエの決心
    1224,  # チェレン
    1121,  # ハイパーボール
    1123,  # ポケモンいれかえ
    1122,  # ポケギア3.0
    13,    # リッチエネルギー
    9,     # ブーメランエネルギー
    11,    # ミストエネルギー
    14,    # スパイクエネルギー
    2,     # 基本【炎】エネルギー
    4,     # 基本【雷】エネルギー
]

CARD_TO_IDX: Dict[int, int] = {card_id: idx for idx, card_id in enumerate(CARD_VOCAB)}

def get_card_index(card_id: int) -> int:
    return CARD_TO_IDX.get(card_id, 0)


class StateEncoder:
    """Encodes PTCG Observation into a fixed-size PyTorch Tensor / NumPy Array

    Input: Observation object from cg.api (or dict equivalent)
    Output: 1D PyTorch Tensor (float32) representing the full game state
    """
    def __init__(self, vocab_size: int = len(CARD_VOCAB), max_bench_slots: int = 5):
        self.vocab_size = vocab_size
        self.max_bench_slots = max_bench_slots

        # Calculation of feature vector dimension
        # 1. Global features (6)
        # 2. Self overview (4) + Hand Histogram (vocab_size)
        # 3. Self Active (2 + vocab_size + 5 status)
        # 4. Self Bench 5 slots * (2 + vocab_size)
        # 5. Opponent overview (4)
        # 6. Opponent Active (2 + vocab_size + 5 status)
        # 7. Opponent Bench 5 slots * (2 + vocab_size)
        
        self.feature_dim = (
            6 + 
            (4 + self.vocab_size) + 
            (2 + self.vocab_size + 5) + 
            self.max_bench_slots * (2 + self.vocab_size) + 
            4 + 
            (2 + self.vocab_size + 5) + 
            self.max_bench_slots * (2 + self.vocab_size)
        )

    def encode(self, obs: Any) -> torch.Tensor:
        """Encodes an Observation instance or dict into a PyTorch Tensor of shape (feature_dim,)"""
        features = []

        state = getattr(obs, "state", None)
        if state is None:
            # Fallback for empty observation
            return torch.zeros(self.feature_dim, dtype=torch.float32)

        your_idx = getattr(state, "yourIndex", 0)
        players = getattr(state, "players", [])
        
        me = players[your_idx] if len(players) > your_idx else None
        opp = players[1 - your_idx] if len(players) > (1 - your_idx) else None

        # -------------------------------------------------------------
        # 1. Global Features (6)
        # -------------------------------------------------------------
        turn = getattr(state, "turn", 0)
        features.append(min(turn / 50.0, 1.0))
        features.append(1.0 if getattr(state, "supporterPlayed", False) else 0.0)
        features.append(1.0 if getattr(state, "stadiumPlayed", False) else 0.0)
        features.append(1.0 if getattr(state, "energyAttached", False) else 0.0)
        features.append(1.0 if getattr(state, "retreated", False) else 0.0)
        features.append(1.0 if getattr(state, "firstPlayer", 0) == your_idx else 0.0)

        # -------------------------------------------------------------
        # 2. Self Player Encoding
        # -------------------------------------------------------------
        if me is not None:
            # Counts
            features.append(getattr(me, "handCount", 0) / 60.0)
            features.append(getattr(me, "deckCount", 0) / 60.0)
            features.append(len(getattr(me, "prize", [])) / 6.0)
            features.append(len(getattr(me, "discard", [])) / 60.0)

            # Hand Card Histogram (One-Hot Counts)
            hand_hist = np.zeros(self.vocab_size, dtype=np.float32)
            hand_cards = getattr(me, "hand", None) or []
            for card in hand_cards:
                cid = getattr(card, "id", 0)
                hand_hist[get_card_index(cid)] += 1.0
            features.extend(hand_hist.tolist())

            # Self Active Pokemon
            active_list = getattr(me, "active", [])
            active_pkm = active_list[0] if active_list and active_list[0] is not None else None
            self._encode_pokemon(active_pkm, features, include_status=True, is_me=True, me_state=me)

            # Self Bench Pokemons (Fixed 5 slots)
            bench_list = getattr(me, "bench", []) or []
            for slot_idx in range(self.max_bench_slots):
                pkm = bench_list[slot_idx] if slot_idx < len(bench_list) else None
                self._encode_pokemon(pkm, features, include_status=False)
        else:
            # Empty fallback padding for Self Player
            features.extend([0.0] * (4 + self.vocab_size + 2 + self.vocab_size + 5 + self.max_bench_slots * (2 + self.vocab_size)))

        # -------------------------------------------------------------
        # 3. Opponent Player Encoding
        # -------------------------------------------------------------
        if opp is not None:
            features.append(getattr(opp, "handCount", 0) / 60.0)
            features.append(getattr(opp, "deckCount", 0) / 60.0)
            features.append(len(getattr(opp, "prize", [])) / 6.0)
            features.append(len(getattr(opp, "discard", [])) / 60.0)

            # Opponent Active Pokemon
            opp_active_list = getattr(opp, "active", [])
            opp_active_pkm = opp_active_list[0] if opp_active_list and opp_active_list[0] is not None else None
            self._encode_pokemon(opp_active_pkm, features, include_status=True, is_me=False, me_state=opp)

            # Opponent Bench Pokemons
            opp_bench_list = getattr(opp, "bench", []) or []
            for slot_idx in range(self.max_bench_slots):
                pkm = opp_bench_list[slot_idx] if slot_idx < len(opp_bench_list) else None
                self._encode_pokemon(pkm, features, include_status=False)
        else:
            # Empty fallback padding for Opponent
            features.extend([0.0] * (4 + 2 + self.vocab_size + 5 + self.max_bench_slots * (2 + self.vocab_size)))

        vec = np.array(features, dtype=np.float32)
        return torch.from_numpy(vec)

    def _encode_pokemon(self, pkm: Any, features: list, include_status: bool = False, is_me: bool = True, me_state: Any = None):
        """Encodes a single Pokemon into feature list"""
        if pkm is None:
            features.append(0.0)  # Exists = 0
            features.append(0.0)  # HP Ratio = 0
            features.extend([0.0] * self.vocab_size)  # One-hot card ID
            if include_status:
                features.extend([0.0] * 5)  # Statuses
            return

        features.append(1.0)  # Exists = 1
        
        # HP ratio
        hp = getattr(pkm, "hp", 0)
        max_hp = getattr(pkm, "maxHp", 1)
        features.append(max(0.0, min(hp / float(max_hp), 1.0)) if max_hp > 0 else 0.0)

        # One-hot card ID
        cid = getattr(pkm, "id", 0)
        card_onehot = [0.0] * self.vocab_size
        card_onehot[get_card_index(cid)] = 1.0
        features.extend(card_onehot)

        # Special conditions (only for Active Pokemon)
        if include_status:
            if me_state is not None:
                features.append(1.0 if getattr(me_state, "poisoned", False) else 0.0)
                features.append(1.0 if getattr(me_state, "burned", False) else 0.0)
                features.append(1.0 if getattr(me_state, "asleep", False) else 0.0)
                features.append(1.0 if getattr(me_state, "paralyzed", False) else 0.0)
                features.append(1.0 if getattr(me_state, "confused", False) else 0.0)
            else:
                features.extend([0.0] * 5)


if __name__ == "__main__":
    encoder = StateEncoder()
    print(f"✅ StateEncoder initialized successfully.")
    print(f"Total Feature Dimension: {encoder.feature_dim}")
