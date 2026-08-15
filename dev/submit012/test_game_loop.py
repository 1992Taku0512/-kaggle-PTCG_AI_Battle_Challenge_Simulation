import sys, os
project_root = os.getcwd()
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dev', 'submit012'))
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))

from dev.common.trainer import PTCGTrainer
from dev.common.config import TrainerConfig
from model import TransformerAlphaZeroNet
from state_encoder import StateEncoder
from mcts import AlphaZeroMCTS

log_file = os.path.join(os.path.dirname(__file__), "test_game_out.txt")
with open(log_file, "w") as f:
    f.write("Initializing trainer...\n")
    deck_path = os.path.join(os.path.dirname(__file__), "deck.csv")
    config = TrainerConfig(
        deck_pool_paths=[deck_path],
        opponent_types=["sampleAgent001"],
        num_episodes=1,
        search_count=5
    )
    trainer = PTCGTrainer(config, TransformerAlphaZeroNet(), StateEncoder(), AlphaZeroMCTS)
    f.write("Starting self_play_game...\n")
    (p1_name, p1_deck), (p2_name, p2_deck) = trainer.deck_provider.sample_deck_pair("fixed", "fixed")
    exps, winner = trainer.collect_self_play_episode(p1_deck, p2_deck, opp_model=None)
    f.write(f"Game finished successfully! winner={winner}, num_experiences={len(exps)}\n")
    print(f"Game finished successfully! winner={winner}, num_experiences={len(exps)}")
