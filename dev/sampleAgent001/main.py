import os
import sys
import random
from typing import List, Any

# Determine absolute path for current agent directory
if "__file__" in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = "/kaggle_simulations/agent"

if not os.path.exists(current_dir):
    current_dir = "/kaggle_simulations/agent"

# Ensure current_dir and cg package are in sys.path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Local dev path fallback
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if os.path.exists(sample_dir) and sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.api import Observation, to_observation_class


def read_deck_csv() -> List[int]:
    """Reads the 60-card deck list for sampleAgent001 with Kaggle path fallbacks."""
    deck_path = os.path.join(current_dir, "deck.csv")
    if not os.path.exists(deck_path):
        deck_path = "/kaggle_simulations/agent/deck.csv"
    if not os.path.exists(deck_path):
        deck_path = "deck.csv"

    with open(deck_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return [int(line.strip()) for line in lines if line.strip() and not line.startswith("#")][:60]


def evaluate_main_option(opt: Any, idx: int) -> float:
    """Master Dragapult ex Dominance Rule Engine with Ultra Ball Priority."""
    opt_type = getattr(opt, "type", None)
    card_id = getattr(opt, "cardId", None)

    score = 100.0 - idx * 0.1

    # 1. ATTACK DECLARATION (type == 12) -> ABSOLUTE MAXIMUM PRIORITY
    if opt_type == 12:
        return 100000000.0

    # 2. DRAGAPULT EX EVOLUTION (121: Stage 2 Dragapult ex, 120: Stage 1 Drakloak)
    elif opt_type == 8 and card_id == 121:
        return 50000000.0  # Stage 2 Dragapult ex (HP 320)
    elif opt_type == 8 and card_id == 120:
        return 30000000.0  # Stage 1 Drakloak

    # 3. ULTRA BALL (1122) & NEST BALL (1121) & SUPPORTERS (1182, 1205)
    elif opt_type == 8 and card_id == 1122:
        return 25000000.0  # Ultra Ball directly fetches Dragapult ex!
    elif opt_type == 8 and card_id == 1121:
        return 20000000.0  # Nest Ball fetches Dreepy!
    elif opt_type == 8 and card_id in (1182, 1205):
        return 15000000.0

    # 4. FIRE (2) & PSYCHIC (5) ENERGY ATTACHMENT FROM HAND
    elif opt_type == 8 and card_id in (2, 5):
        return 10000000.0 if card_id == 2 else 9000000.0

    # 5. DREEPY (119) & BASIC ATTACKERS (Ugatsuhomura 46, Drampa 1010)
    elif opt_type == 8 and card_id in (119, 46, 1010):
        return 5000000.0

    # 6. OTHER PLAYABLE CARDS FROM HAND
    elif opt_type == 8:
        return 100000.0

    # 7. END TURN (type == 14) -> LOWEST PRIORITY
    elif opt_type == 14:
        return 1.0

    return score


def evaluate_sub_option(opt: Any, idx: int) -> float:
    """Sub-prompt Engine: Direct Placement on Active Slot (Dragapult line)."""
    card_id = getattr(opt, "cardId", None)
    in_play_area = getattr(opt, "inPlayArea", None)

    score = 100.0 - idx * 0.1

    # Prefer options pointing to Active Area (inPlayArea 0 or 4)
    if in_play_area in (0, 4):
        score += 5000000.0
    elif in_play_area is not None:
        score += 1000000.0

    # Pick Dragapult ex line & Energies from deck search
    if card_id in (121, 120, 119):
        score += 3000000.0
    elif card_id in (2, 5):
        score += 2000000.0

    return score


def agent(obs_dict: Any) -> List[int]:
    """Master Dragapult ex Rule Agent.
    
    Fully compliant with Kaggle competition format.
    """
    obs: Observation = to_observation_class(obs_dict)

    # Initial selection: Return 60-card deck
    if obs.select is None:
        return read_deck_csv()

    options = getattr(obs.select, "option", [])
    if not options:
        return [0]

    num_opts = len(options)
    min_cnt = max(1, getattr(obs.select, "minCount", 1))

    # SUB-PROMPT SELECTION PHASE
    has_end_turn_option = any(getattr(o, "type", None) == 14 for o in options)
    if not has_end_turn_option:
        scored_sub = [(evaluate_sub_option(opt, i), i) for i, opt in enumerate(options)]
        scored_sub.sort(key=lambda x: x[0], reverse=True)
        target_cnt = min(min_cnt, num_opts)
        selected_sub = [idx for _, idx in scored_sub[:target_cnt]]
        return selected_sub if selected_sub else list(range(min(min_cnt, num_opts)))

    # MAIN ACTION SELECTION PHASE
    scored_options = [(evaluate_main_option(opt, i), i) for i, opt in enumerate(options)]
    scored_options.sort(key=lambda x: x[0], reverse=True)

    # Pick top min_cnt options
    target_cnt = min(min_cnt, num_opts)
    selected_indices = [idx for _, idx in scored_options[:target_cnt]]

    # Strict bounds checking
    valid_indices = [i for i in selected_indices if 0 <= i < num_opts]
    if len(valid_indices) < min_cnt:
        valid_indices = list(range(min(min_cnt, num_opts)))

    return valid_indices
