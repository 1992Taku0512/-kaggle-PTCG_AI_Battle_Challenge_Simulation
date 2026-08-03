import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit003.model import AlphaZeroNet
from dev.submit003.state_encoder import StateEncoder


def main():
    # 1. Setup Config (Dynamic Decks + Dynamic Opponents + Resume/Fresh)
    config = TrainerConfig(
        experiment_name="example_trainer_run",
        
        # Fresh training (Set to a .pt path if resuming/fine-tuning)
        resume_checkpoint_path=None,  
        
        # Dynamic Opponent Mode: Self-Play (70%), Random Agent (15%), Past Checkpoints (15%)
        opponent_types=["self_play", "random", "past_checkpoint"],
        opponent_weights=[0.7, 0.15, 0.15],
        past_checkpoint_dir=os.path.join(project_root, "checkpoints"),
        
        # Dynamic Deck Mode: Randomly sample P1 and P2 decks from dev/deck_pool/
        p1_deck_mode="random",
        p2_deck_mode="random",
        
        # Hyperparameters
        num_episodes=500,
        batch_size=64,
        eval_every=100,
        eval_num_games=10,
        checkpoint_dir="checkpoints/example_run",
        use_line_notify=False
    )

    # 2. Instantiate Model and Encoder (Model is passed as a parameter to Trainer!)
    model = AlphaZeroNet()
    encoder = StateEncoder()

    # 3. Create Trainer Wrapper
    trainer = PTCGTrainer(
        config=config,
        model=model,        # <--- Passed as parameter! Easy to swap models in future!
        encoder=encoder,
        model_cls=AlphaZeroNet
    )

    print("Trainer initialized successfully!")
    print(f"Loaded Decks in Pool: {len(trainer.deck_provider.decks)}")
    
    # Sample a deck pair test
    p1, p2 = trainer.deck_provider.sample_deck_pair("random", "random")
    print(f"Sampled P1 Deck: {p1[0]} | P2 Deck: {p2[0]}")
    
    # Sample opponent test
    opp_type = trainer.opponent_provider.sample_opponent_type()
    opp_desc, opp_model = trainer.opponent_provider.get_opponent_model(opp_type, model)
    print(f"Sampled Opponent Type: {opp_desc}")


if __name__ == "__main__":
    main()
