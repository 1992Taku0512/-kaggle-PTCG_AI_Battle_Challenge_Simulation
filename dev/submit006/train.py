import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit006_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit006_dir not in sys.path:
    sys.path.insert(0, submit006_dir)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit006.model import TransformerAlphaZeroNet
from dev.submit006.state_encoder import StateEncoder
from dev.submit006.mcts import AlphaZeroMCTS


def main():
    print("=" * 80)
    print("🚀 Launching dev/submit006 Multi-Opponent Practical Training (10,000 Additional Episodes)")
    print("   Mode: MULTI-OPPONENT (Self-Play 60% | Random 20% | Past 20%)")
    print("   P1 & P2 Decks: FIXED (dev/submit006/deck.csv)")
    print("   MCTS Search Count: 100 per turn")
    print("   Sliding Window: Recent 100 Games Winrate")
    print("   LINE Notification: Every 2,500 Episodes")
    print("=" * 80)

    latest_ckpt = os.path.join(submit006_dir, "checkpoints", "latest_model.pt")
    resume_path = latest_ckpt if os.path.exists(latest_ckpt) else None

    # Config for 10,000 Additional Episode Multi-Opponent Training (Ep 11001 -> 21000)
    config = TrainerConfig(
        experiment_name="submit006_multiopp_10000ep",
        resume_checkpoint_path=resume_path,  # Resume from trained weights (Ep 11000)
        
        # Multi-Opponent Diversity (Self-Play 60%, RandomAgent 20%, Past Checkpoint 20%)
        opponent_types=["self_play", "random", "past_checkpoint"],
        opponent_weights=[0.60, 0.20, 0.20],
        
        # Fixed Decks (P1 & P2 both fixed to dev/submit006/deck.csv)
        p1_deck_mode="fixed",
        p2_deck_mode="fixed",
        
        # Hyperparameters
        num_episodes=21000,  # Ep 11001 -> Ep 21000 (10,000 additional episodes)
        batch_size=64,
        lr=1e-3,
        search_count=100,  # 100 simulations per turn
        
        # Evaluation & LINE Notification (Recent 100 games winrate, every 2,500 episodes)
        recent_winrate_window=100,
        eval_every=2500,
        eval_num_games=20,
        checkpoint_dir=os.path.join(submit006_dir, "checkpoints"),
        past_checkpoint_dir=os.path.join(submit006_dir, "checkpoints"),
        use_line_notify=True,
        line_notify_every=2500
    )

    model = TransformerAlphaZeroNet(d_model=256, num_heads=4, d_feedforward=512, num_layers_encoder=2, num_layers_decoder=2)
    encoder = StateEncoder()

    trainer = PTCGTrainer(
        config=config,
        model=model,
        encoder=encoder,
        mcts_cls=AlphaZeroMCTS,
        model_cls=TransformerAlphaZeroNet
    )

    trainer.train(agent_save_dir="dev/submit006")
    print("\n🎉 10,000 Additional Episode Multi-Opponent Training Completed for dev/submit006!")


if __name__ == "__main__":
    main()
