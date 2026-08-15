import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit010_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit010_dir not in sys.path:
    sys.path.insert(0, submit010_dir)

# cg パッケージへのパス
cg_path = os.path.join(submit010_dir, "cg")
if cg_path not in sys.path:
    sys.path.insert(0, cg_path)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit010.model import TransformerAlphaZeroNet
from dev.submit010.state_encoder import StateEncoder
from dev.submit010.mcts import AlphaZeroMCTS


def main():
    submit010_deck = os.path.join(submit010_dir, "deck.csv")
    checkpoint_dir = os.path.join(submit010_dir, "checkpoints")

    config = TrainerConfig(
        experiment_name="submit010_overnight_30000ep_report_aligned",
        resume_checkpoint_path=os.path.join(checkpoint_dir, "latest_model.pt"),

        # Opponent Mix (40% sampleAgent002 / 30% submit009 / 20% sampleAgent001 / 10% self_play)
        opponent_types=["sampleAgent002", "submit009", "sampleAgent001", "self_play"],
        opponent_weights=[0.4, 0.3, 0.2, 0.1],

        # 超高火力ハイブリッドデッキ
        deck_pool_paths=[submit010_deck],
        p1_deck_mode="fixed",
        p2_deck_mode="fixed",

        # 中間報酬 (Reward Shaping v3 - 分析レポート完全準拠)
        use_intermediate_reward=True,
        reward_side_take=0.3,       # サイド取得 +0.3
        reward_side_lost=-0.2,      # サイド被取 -0.2
        reward_attack_use=0.1,      # 攻撃宣言 +0.1

        # Hyperparameters
        num_episodes=30000,         # 30,000 エピソード長時学習
        batch_size=64,
        lr=3e-4,
        search_count=150,           # MCTS シミュレーション 150/ターン

        # Winrate window: 直近100試合
        recent_winrate_window=100,

        # Checkpoint 保存: 500エピソードごと
        save_checkpoint_every=500,
        checkpoint_dir=checkpoint_dir,

        # LINE通知: 2,500エピソードごと
        line_notify_every=2500
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
    print(f"🚀 Starting Overnight Production RL Training: {config.experiment_name}")
    print(f"• Total Episodes: {config.num_episodes}")
    print(f"• Opponents: {config.opponent_types} with weights {config.opponent_weights}")
    print(f"• Checkpoint Dir: {checkpoint_dir}")
    print(f"• LINE Notify: Every {config.line_notify_every} episodes")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
