import os
import sys
import torch

if "__file__" in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = "/kaggle_simulations/agent" if os.path.exists("/kaggle_simulations/agent") else os.getcwd()

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Local project cg path fallback (for local eval)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
cg_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if os.path.exists(cg_dir) and cg_dir not in sys.path:
    sys.path.insert(0, cg_dir)

from cg.api import to_observation_class
from state_encoder import StateEncoder
from model import TransformerAlphaZeroNet
from mcts import AlphaZeroMCTS

# Initialize model and agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerAlphaZeroNet().to(device)
encoder = StateEncoder()

weights_path = os.path.join(current_dir, "model_weights.pt")
if not os.path.exists(weights_path):
    weights_path = os.path.join(os.getcwd(), "model_weights.pt")

if os.path.exists(weights_path):
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print("Loaded submit010 model weights successfully.")
    except Exception as e:
        print(f"Warning: Failed to load model weights: {e}")

mcts_engine = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=150, device=device)


def read_deck_csv() -> list[int]:
    """Reads 60 card IDs from deck.csv."""
    deck_path = os.path.join(current_dir, "deck.csv")
    if not os.path.exists(deck_path):
        deck_path = "deck.csv"
    if os.path.exists(deck_path):
        with open(deck_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        cards = [int(line.strip()) for line in lines if line.strip() and not line.startswith("#")]
        if len(cards) == 60:
            return cards
    return [46] * 60


def agent(obs_dict: dict, context: dict = None) -> list[int]:
    """Main Kaggle agent entry point called on every turn."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    if not obs.select.option:
        return []

    num_opts = len(obs.select.option)
    min_cnt = max(1, getattr(obs.select, "minCount", 1))

    my_deck = read_deck_csv()
    try:
        action_list, _, _ = mcts_engine.get_action_distribution(obs, your_deck=my_deck)
        if isinstance(action_list, list):
            valid_actions = [a for a in action_list if 0 <= a < num_opts]
            if len(valid_actions) >= min_cnt:
                return valid_actions[:min_cnt]
    except Exception as e:
        pass

    return list(range(min(min_cnt, num_opts)))
