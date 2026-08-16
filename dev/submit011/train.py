import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit011_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit011_dir not in sys.path:
    sys.path.insert(0, submit011_dir)

cg_path = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if cg_path not in sys.path:
    sys.path.insert(0, cg_path)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from model import TransformerAlphaZeroNet
from state_encoder import StateEncoder
from mcts import AlphaZeroMCTS


def main():
    submit011_deck = os.path.join(submit011_dir, "deck.csv")
    checkpoint_dir = os.path.join(submit011_dir, "checkpoints")

    config = TrainerConfig(
        experiment_name="submit011_prioritized_mcts_water_hybrid_10000ep",
        resume_checkpoint_path=None,

        # Opponent Mix (30% submit009 / 30% sampleAgent001 / 20% sampleAgent002 / 20% official_sample)
        opponent_types=["submit009", "sampleAgent001", "sampleAgent002", "official_sample"],
        opponent_weights=[0.3, 0.3, 0.2, 0.2],

        deck_pool_paths=[submit011_deck],
        p1_deck_mode="fixed",
        p2_deck_mode="fixed",

        # 中間報酬 (Reward Shaping v4)
        use_intermediate_reward=True,
        reward_side_take=0.3,
        reward_side_lost=-0.2,
        reward_attack_use=0.1,

        # Hyperparameters (学習時: MCTS 200 simulations)
        num_episodes=10000,
        batch_size=64,
        lr=3e-4,
        search_count=200,

        recent_winrate_window=100,
        save_checkpoint_every=500,
        checkpoint_dir=checkpoint_dir,
        line_notify_every=1000
    )

    encoder = StateEncoder()
    model = TransformerAlphaZeroNet()

    trainer = PTCGTrainer(
        config=config,
        model=model,
        encoder=encoder,
        mcts_cls=AlphaZeroMCTS
    )

    print("=" * 80)
    print(f"🚀 Starting submit011 Prioritized MCTS RL Training: {config.experiment_name}")
    print(f"• Deck: submit011 24-Water Deck (Boss Orders + Switch + Ultra Ball)")
    print(f"• Training Search Count: {config.search_count} simulations (Prioritized MCTS)")
    print(f"• Opponents: {config.opponent_types} with weights {config.opponent_weights}")
    print(f"• Checkpoint Dir: {checkpoint_dir}")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
