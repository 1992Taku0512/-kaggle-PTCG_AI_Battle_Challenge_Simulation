import os
import sys

# Ensure Kaggle simulation agent directory and local import paths are setup
kaggle_agent_dir = "/kaggle_simulations/agent"
if os.path.exists(kaggle_agent_dir) and kaggle_agent_dir not in sys.path:
    sys.path.insert(0, kaggle_agent_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

sample_dir = os.path.abspath("data/sample_submission/sample_submission")
if os.path.exists(sample_dir) and sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

import torch
from cg.api import (
    Observation, SelectType, OptionType, SelectContext, AreaType,
    to_observation_class
)

from state_encoder import StateEncoder
from model import AlphaZeroNet
from mcts import AlphaZeroMCTS

# Global Instances for Agent Inference Efficiency
_ENCODER = None
_MODEL = None
_MCTS = None

def get_agent_components():
    global _ENCODER, _MODEL, _MCTS
    if _MCTS is None:
        _ENCODER = StateEncoder()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = AlphaZeroNet().to(device)
        
        # Load weights if available
        weights_path = os.path.join(current_dir, "model_weights.pt")
        if not os.path.exists(weights_path):
            weights_path = "/kaggle_simulations/agent/model_weights.pt"
        if os.path.exists(weights_path):
            try:
                _MODEL.load_state_dict(torch.load(weights_path, map_location=device))
                print(f"Loaded trained weights from: {weights_path}")
            except Exception as e:
                print(f"Failed to load weights ({e}), using initialized model.")
                
        _MODEL.eval()
        _MCTS = AlphaZeroMCTS(model=_MODEL, encoder=_ENCODER, num_simulations=40, device=device)
    return _ENCODER, _MODEL, _MCTS


def read_deck_csv() -> list[int]:
    """Read deck.csv (60 card IDs)."""
    file_path = os.path.join(current_dir, "deck.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    if not os.path.exists(file_path):
        file_path = "deck.csv"
    with open(file_path, "r") as file:
        csv = file.read().strip().split("\n")
    deck = [int(line.strip()) for line in csv if line.strip()][:60]
    return deck


def agent(obs_dict: dict) -> list[int]:
    """AlphaZero Deep Reinforcement Learning Agent for Kaggle PTCG AI Battle."""
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()

        options = obs.select.option
        num_options = len(options)
        min_count = obs.select.minCount
        max_count = obs.select.maxCount
        ctx = obs.select.context
        sel_type = obs.select.type

        # 0 options case
        if num_options == 0:
            if sel_type == SelectType.MAIN or ctx == SelectContext.MAIN:
                return [0]
            if obs.select.deck is not None and len(obs.select.deck) > 0:
                k = min(min_count, len(obs.select.deck)) if min_count > 0 else 0
                return list(range(k))
            return []

        # 1. MAIN Turn Selection -> AlphaZero MCTS Engine!
        if sel_type == SelectType.MAIN or ctx == SelectContext.MAIN:
            encoder, model, mcts_engine = get_agent_components()
            action_indices, probs, _ = mcts_engine.get_action_distribution(obs)
            if action_indices and 0 <= action_indices[0] < num_options:
                return action_indices
            return [0]

        # 2. Boolean / Setup Decisions
        if sel_type == SelectType.YES_NO or ctx in (SelectContext.IS_FIRST, SelectContext.MULLIGAN, SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT):
            return [0]

        # 3. Setup / Pokemon placement (Active / Bench / Switch)
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_ACTIVE, SelectContext.TO_BENCH, SelectContext.SWITCH):
            card_db = get_card_data_dict()
            for idx, opt in enumerate(options):
                cid = getattr(opt, 'cardId', None) if not isinstance(opt, dict) else opt.get('cardId')
                if cid and cid in card_db:
                    cdata = card_db[cid]
                    if getattr(cdata, 'cardType', None) == 0 or getattr(cdata, 'basic', False):
                        return [idx]
            return [0]

        # 4. Count selection
        if sel_type == SelectType.COUNT:
            return [min_count] if min_count <= num_options else [0]

        # 5. Generic Option Filter (minCount == 0 -> Pass)
        if min_count == 0:
            return []

        # Required multi-selection (minCount > 0)
        k = min(min_count, num_options)
        return list(range(k)) if k > 0 else []

    except Exception:
        # Multi-layered Safe Fallback Guarantee
        try:
            select_info = obs_dict.get("select", {})
            opts = select_info.get("option", [])
            min_c = select_info.get("minCount", 0)
            if not opts or min_c == 0:
                return []
            k = min(max(min_c, 1), len(opts))
            return list(range(k))
        except Exception:
            return []
            return list(range(min(max(min_c, 1), len(opts))))
        except Exception:
            return []
