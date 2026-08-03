import os
import sys

# ---- Path Setup (Kaggle compatible) ----
# On Kaggle: __file__ is not defined, agent files live at /kaggle_simulations/agent/
# Locally: __file__ is defined, agent files live alongside this script
kaggle_agent_dir = "/kaggle_simulations/agent"
if os.path.exists(kaggle_agent_dir) and kaggle_agent_dir not in sys.path:
    sys.path.insert(0, kaggle_agent_dir)

try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR = kaggle_agent_dir if os.path.exists(kaggle_agent_dir) else os.getcwd()

if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

# Local dev: search for cg module in several possible locations
for _candidate in [
    os.path.join(os.getcwd(), "data", "sample_submission", "sample_submission"),
    os.path.join(_AGENT_DIR, "..", "..", "data", "sample_submission", "sample_submission"),
    os.path.join(_AGENT_DIR, "..", "..", "..", "data", "sample_submission", "sample_submission"),
]:
    _candidate = os.path.normpath(_candidate)
    if os.path.isdir(os.path.join(_candidate, "cg")) and _candidate not in sys.path:
        sys.path.append(_candidate)  # append so Kaggle's own cg takes priority
        break

# ---- Imports ----
from cg.api import (
    Observation, SelectType, OptionType, SelectContext, AreaType,
    to_observation_class, all_card_data
)

# ---- Lazy-loaded heavy modules ----
_torch = None
_StateEncoder = None
_AlphaZeroNet = None
_AlphaZeroMCTS = None


def _lazy_import_heavy():
    """Import torch and model modules on first use (not at module load time)."""
    global _torch, _StateEncoder, _AlphaZeroNet, _AlphaZeroMCTS
    if _torch is None:
        import torch as _t
        _torch = _t
        from state_encoder import StateEncoder as _SE
        from model import AlphaZeroNet as _AZN
        from mcts import AlphaZeroMCTS as _AZMCTS
        _StateEncoder = _SE
        _AlphaZeroNet = _AZN
        _AlphaZeroMCTS = _AZMCTS


# ---- Card Data Cache ----
_CARD_DATA_CACHE = None

def get_card_data_dict():
    global _CARD_DATA_CACHE
    if _CARD_DATA_CACHE is None:
        cards = all_card_data()
        _CARD_DATA_CACHE = {c.cardId: c for c in cards}
    return _CARD_DATA_CACHE


# ---- AlphaZero Components (singleton) ----
_ENCODER = None
_MODEL = None
_MCTS = None

def get_agent_components():
    global _ENCODER, _MODEL, _MCTS
    if _MCTS is None:
        _lazy_import_heavy()
        _ENCODER = _StateEncoder()
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _MODEL = _AlphaZeroNet().to(device)

        # Search for weights file
        for wpath in [
            os.path.join(_AGENT_DIR, "model_weights.pt"),
            "/kaggle_simulations/agent/model_weights.pt",
            "model_weights.pt",
        ]:
            if os.path.exists(wpath):
                try:
                    _MODEL.load_state_dict(_torch.load(wpath, map_location=device))
                    print(f"Loaded trained weights from: {wpath}")
                except Exception as e:
                    print(f"Failed to load weights from {wpath}: {e}")
                break

        _MODEL.eval()
        _MCTS = _AlphaZeroMCTS(model=_MODEL, encoder=_ENCODER, num_simulations=40, device=device)
    return _ENCODER, _MODEL, _MCTS


# ---- Deck Reader ----
def read_deck_csv() -> list[int]:
    """Read deck.csv (60 card IDs)."""
    for fpath in [
        os.path.join(_AGENT_DIR, "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
        "deck.csv",
    ]:
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                lines = f.read().strip().split("\n")
            return [int(line.strip()) for line in lines if line.strip()][:60]
    raise FileNotFoundError("deck.csv not found")


# ---- Agent Entry Point ----
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

        # 1. MAIN Turn Selection -> AlphaZero MCTS Engine
        if sel_type == SelectType.MAIN or ctx == SelectContext.MAIN:
            encoder, model, mcts_engine = get_agent_components()
            action_indices, probs, _ = mcts_engine.get_action_distribution(obs)
            if action_indices and 0 <= action_indices[0] < num_options:
                return action_indices
            return [0]

        # 2. Boolean / Setup Decisions
        if sel_type == SelectType.YES_NO or ctx in (
            SelectContext.IS_FIRST, SelectContext.MULLIGAN,
            SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT
        ):
            return [0]

        # 3. Setup / Pokemon placement (Active / Bench / Switch)
        if ctx in (
            SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
            SelectContext.TO_ACTIVE, SelectContext.TO_BENCH, SelectContext.SWITCH
        ):
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
        # Emergency Fallback
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
