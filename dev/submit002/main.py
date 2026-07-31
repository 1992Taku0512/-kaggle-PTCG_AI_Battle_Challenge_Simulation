import os
import sys

# Ensure /kaggle_simulations/agent is in sys.path
kaggle_agent_dir = "/kaggle_simulations/agent"
if os.path.exists(kaggle_agent_dir) and kaggle_agent_dir not in sys.path:
    sys.path.insert(0, kaggle_agent_dir)


sample_dir = os.path.abspath("data/sample_submission/sample_submission")
if os.path.exists(sample_dir) and sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.api import (
    Observation, SelectType, OptionType, SelectContext, AreaType,
    to_observation_class, all_card_data
)


_CARD_DATA_CACHE = None

def get_card_data_dict():
    global _CARD_DATA_CACHE
    if _CARD_DATA_CACHE is None:
        cards = all_card_data()
        _CARD_DATA_CACHE = {c.cardId: c for c in cards}
    return _CARD_DATA_CACHE

def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck



def agent(obs_dict: dict) -> list[int]:
    """100% Rule-compliant PTCG Agent without IndexError."""
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
        card_db = get_card_data_dict()
        
        # When option list is empty
        if num_options == 0:
            if sel_type == SelectType.MAIN or ctx == SelectContext.MAIN:
                return [0]
            if obs.select.deck is not None and len(obs.select.deck) > 0:
                k = min(min_count, len(obs.select.deck)) if min_count > 0 else 0
                return list(range(k))
            return []

        
        # 1. MAIN Turn Selection
        if sel_type == SelectType.MAIN or ctx == SelectContext.MAIN:
            priority_order = [
                OptionType.ATTACK,
                OptionType.PLAY,
                OptionType.EVOLVE,
                OptionType.ABILITY,
                OptionType.END
            ]
            for preferred_type in priority_order:
                for idx, opt in enumerate(options):
                    if opt.type == preferred_type:
                        return [idx]
            return [0]

        
        # 2. Boolean / Setup Decisions (YES_NO, IS_FIRST, MULLIGAN, etc.)
        if sel_type == SelectType.YES_NO or ctx in (SelectContext.IS_FIRST, SelectContext.MULLIGAN, SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT):
            return [0]
            
        # 3. Setup & Pokemon selection (Active / Bench)
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_ACTIVE, SelectContext.TO_BENCH, SelectContext.SWITCH):
            # Prioritize basic Pokemon cards
            for idx, opt in enumerate(options):
                cid = getattr(opt, 'cardId', None)
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
        if k <= 0:
            return []
            
        return list(range(k))

    except Exception:
        # Emergency Fallback
        try:
            select_info = obs_dict.get("select", {})
            opts = select_info.get("option", [])
            deck_opts = select_info.get("deck", [])
            num_opts = len(opts)
            min_c = select_info.get("minCount", 0)
            if num_opts == 0:
                if deck_opts and len(deck_opts) > 0 and min_c > 0:
                    return list(range(min(min_c, len(deck_opts))))
                return []
            if min_c == 0:
                return []
            k = min(max(min_c, 1), num_opts)
            return list(range(k))
        except Exception:
            return []


