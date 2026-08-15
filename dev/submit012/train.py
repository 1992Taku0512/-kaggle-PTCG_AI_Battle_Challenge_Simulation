import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit012_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit012_dir not in sys.path:
    sys.path.insert(0, submit012_dir)

cg_path = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if cg_path not in sys.path:
    sys.path.insert(0, cg_path)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from model import TransformerAlphaZeroNet
from state_encoder import StateEncoder
from mcts import AlphaZeroMCTS


def get_threshold_for_episode(ep: int) -> float:
    """Returns the curriculum min_p_threshold for the given episode number."""
    if ep <= 20000:
        return 0.0         # Phase 1: 0.0 (No pruning)
    elif ep <= 40000:
        return 0.0025      # Phase 2: 0.25%
    elif ep <= 60000:
        return 0.005       # Phase 3: 0.50%
    elif ep <= 80000:
        return 0.0075      # Phase 4: 0.75%
    else:
        return 0.01        # Phase 5: 1.0% (Standard Upper Limit)


def main():
    submit012_deck = os.path.join(submit012_dir, "deck.csv")
    checkpoint_dir = os.path.join(submit012_dir, "checkpoints")
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_model.pt")
    
    resume_path = latest_checkpoint if os.path.exists(latest_checkpoint) else None

    config = TrainerConfig(
        experiment_name="submit012_100k_curriculum_rl_v7",
        resume_checkpoint_path=resume_path,

        # Opponent Mix (30% submit009 / 30% sampleAgent001 / 20% sampleAgent002 / 20% official_sample)
        opponent_types=["submit009", "sampleAgent001", "sampleAgent002", "official_sample"],
        opponent_weights=[0.3, 0.3, 0.2, 0.2],

        deck_pool_paths=[submit012_deck],
        p1_deck_mode="fixed",
        p2_deck_mode="fixed",

        use_intermediate_reward=True,
        reward_side_take=0.30,
        reward_side_lost=-0.20,
        reward_attack_use=0.10,

        num_episodes=100000,
        batch_size=64,
        lr=3e-4,
        search_count=200,

        recent_winrate_window=100,
        save_checkpoint_every=10000,
        checkpoint_dir=checkpoint_dir,
        line_notify_every=10000       # LINE Messaging API 節約のため 10,000 エピソードごとに通知（全体で10通のみ）
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
    print(f"🚀 Resuming submit012 100,000 Episode Automated Curriculum Training: {config.experiment_name}")
    print(f"• Resuming from: {resume_path}")
    print(f"• Deck: submit012 11-Basic Pokemon 25-Water Deck (Boss Orders + Switch + Ultra Ball)")
    print(f"• Reward Shaping: v7 Clean Design (LO Penalty -2.0, Energy Readiness Shaping, Damage Scale)")
    print(f"• MCTS Config: Prioritized MCTS (Curriculum 0.0 -> 0.01 across 5 Phases)")
    print(f"• Training Search Count: {config.search_count} simulations")
    print(f"• Checkpoint Step: Every {config.save_checkpoint_every} episodes")
    print(f"• LINE Notify Step: Every {config.line_notify_every} episodes")
    print(f"• Checkpoint Dir: {checkpoint_dir}")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
