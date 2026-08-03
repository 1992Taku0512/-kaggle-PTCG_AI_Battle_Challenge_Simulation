import os
import sys

# Ensure project root and submit005 are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit005_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit005_dir not in sys.path:
    sys.path.insert(0, submit005_dir)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit005.model import AlphaZeroNet
from dev.submit005.state_encoder import StateEncoder
from dev.submit005.mcts import AlphaZeroMCTS


def main():
    print("=" * 80)
    print("🚀 Launching dev/submit005 Training with Generic PTCGTrainer Wrapper")
    print("   Mode: CLEAN INITIALIZATION (Fresh New Model Weights)")
    print("   Target Episodes: 2,000")
    print("=" * 80)

    # 1. Config Setup
    config = TrainerConfig(
        experiment_name="submit005_wrapper_2000ep",
        
        # Fresh initialization (No resume path)
        resume_checkpoint_path=None,
        
        # Dynamic Opponents (70% Self-Play, 15% Random, 15% Past Checkpoint)
        opponent_types=["self_play", "random", "past_checkpoint"],
        opponent_weights=[0.7, 0.15, 0.15],
        past_checkpoint_dir=os.path.join(project_root, "checkpoints"),
        
        # Dynamic Decks (Random selection for both P1 and P2 from dev/deck_pool)
        p1_deck_mode="random",
        p2_deck_mode="random",
        
        # Hyperparameters
        num_episodes=2000,
        batch_size=64,
        lr=1e-3,
        search_count=20,  # Fast & effective MCTS search count for 2000 episodes
        eval_every=500,
        eval_num_games=20,
        checkpoint_dir=os.path.join(submit005_dir, "checkpoints_wrapper"),
        use_line_notify=True,
        line_notify_every=500
    )

    # 2. Instantiate submit005 Model & Encoder
    model = AlphaZeroNet(state_dim=269, hidden_dim=512, action_dim=64, num_res_blocks=2)
    encoder = StateEncoder()

    # 3. Create Trainer Wrapper (Model is passed as a parameter!)
    trainer = PTCGTrainer(
        config=config,
        model=model,
        encoder=encoder,
        mcts_cls=AlphaZeroMCTS,
        model_cls=AlphaZeroNet
    )

    # 4. Start 2000 Episode Training Run
    trainer.train(agent_save_dir="dev/submit005")


if __name__ == "__main__":
    main()
