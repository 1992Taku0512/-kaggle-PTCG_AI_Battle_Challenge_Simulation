import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit007_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit007_dir not in sys.path:
    sys.path.insert(0, submit007_dir)

# cg パッケージへのパス
cg_path = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if cg_path not in sys.path:
    sys.path.insert(0, cg_path)

from dev.common.config import TrainerConfig
from dev.common.trainer import PTCGTrainer
from dev.submit007.model import TransformerAlphaZeroNet
from dev.submit007.state_encoder import StateEncoder
from dev.submit007.mcts import AlphaZeroMCTS


def main():
    submit007_deck = os.path.join(submit007_dir, "deck.csv")
    checkpoint_dir = os.path.join(submit007_dir, "checkpoints")

    config = TrainerConfig(
        experiment_name="submit007_dragapult_vs_sampleAgent001_10000ep",
        resume_checkpoint_path=None,  # 新規モデルでスタート

        # Opponent Mix (50% 自己対局 / 30% sampleAgent001 強敵 / 20% 公式サンプル)
        opponent_types=["self_play", "sampleAgent001", "official_sample"],
        opponent_weights=[0.5, 0.3, 0.2],

        # Dragapult ex デッキ固定
        deck_pool_paths=[submit007_deck],
        p1_deck_mode="fixed",
        p2_deck_mode="fixed",

        # バランス型中間報酬（Reward Shaping v2）有効化
        use_intermediate_reward=True,
        reward_side_take=0.3,       # サイド取得 +0.3
        reward_side_lost=-0.2,      # サイド被取 -0.2
        reward_stage2_evolve=0.2,   # ドラパルトex (2進化) +0.2
        reward_stage1_evolve=0.1,   # ドロンチ (1進化) +0.1
        reward_attack_use=0.05,     # 攻撃宣言 (技発動) +0.05

        # Training Hyperparameters
        num_episodes=10000,        # 本番: 10,000 エピソード
        batch_size=64,
        lr=3e-4,                   # 安定学習率 3e-4
        search_count=100,          # 100 MCTS simulations/turn

        # Winrate window: 直近100試合
        recent_winrate_window=100,

        # Checkpoint 保存: 500エピソードごと
        save_checkpoint_every=500,
        checkpoint_dir=checkpoint_dir,

        # 途中の評価対局は無効化 (完走時にまとめて対戦実行)
        eval_every=100000,
        eval_num_games=20,

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
    print(f"🚀 Starting Production Training: {config.experiment_name}")
    print(f"• Opponents: {config.opponent_types} with weights {config.opponent_weights}")
    print(f"• Episodes: {config.num_episodes}")
    print(f"• Reward Shaping: Side(+{config.reward_side_take}/-{abs(config.reward_side_lost)}), Stage2(+{config.reward_stage2_evolve}), Stage1(+{config.reward_stage1_evolve}), Attack(+{config.reward_attack_use})")
    print(f"• Checkpoint Dir: {checkpoint_dir}")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
