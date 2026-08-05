import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit008_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit008_dir not in sys.path:
    sys.path.insert(0, submit008_dir)

# cg パッケージへのパス
cg_path = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if cg_path not in sys.path:
    sys.path.insert(0, cg_path)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit008.model import TransformerAlphaZeroNet
from dev.submit008.state_encoder import StateEncoder
from dev.submit008.mcts import AlphaZeroMCTS


def main():
    submit008_deck = os.path.join(submit008_dir, "deck.csv")
    checkpoint_dir = os.path.join(submit008_dir, "checkpoints")

    config = TrainerConfig(
        experiment_name="submit008_fast_fire_beatdown_10000ep",
        resume_checkpoint_path=None,

        # Opponent Mix (40% sampleAgent001 / 30% official_sample / 20% submit005 / 10% self_play)
        opponent_types=["sampleAgent001", "official_sample", "self_play"],
        opponent_weights=[0.5, 0.3, 0.2],

        # 高速火ビートダウンデッキ
        deck_pool_paths=[submit008_deck],
        p1_deck_mode="fixed",
        p2_deck_mode="fixed",

        # 中間報酬 (Reward Shaping v2)
        use_intermediate_reward=True,
        reward_side_take=0.3,       # サイド取得 +0.3
        reward_side_lost=-0.2,      # サイド被取 -0.2
        reward_attack_use=0.1,      # 攻撃宣言 (技発動) +0.1

        # Hyperparameters
        num_episodes=10000,
        batch_size=64,
        lr=3e-4,
        search_count=100,

        # Winrate window: 直近100試合
        recent_winrate_window=100,

        # Checkpoint 保存: 500エピソードごと
        save_checkpoint_every=500,
        checkpoint_dir=checkpoint_dir,

        # LINE通知: 1,000エピソードごと
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
    print(f"🚀 Starting High-Speed Fire Beatdown Training: {config.experiment_name}")
    print(f"• Target Goal: 70%+ Winrate against all benchmark models before 20:00")
    print(f"• Opponents: {config.opponent_types} with weights {config.opponent_weights}")
    print(f"• Checkpoint Dir: {checkpoint_dir}")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
