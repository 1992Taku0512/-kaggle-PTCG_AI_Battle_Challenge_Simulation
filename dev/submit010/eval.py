"""
submit010 評価スクリプト
submit010 (latest_model.pt) vs sampleAgent001, sampleAgent002, submit009
各対戦相手に対して 20 試合ずつ評価
"""
import os
import sys
import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit010_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit010_dir not in sys.path:
    sys.path.insert(0, submit010_dir)

# cg パッケージは sample_submission から読む（trainer.py に合わせる）
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)


def read_deck_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return [int(l.strip()) for l in lines if l.strip() and not l.startswith("#")][:60]


from dev.common.opponent_provider import (
    OpponentProvider, agent_sample001, agent_sample002, agent_submit009
)

from cg.game import battle_start, battle_select
from cg.api import to_observation_class
from state_encoder import StateEncoder
from model import TransformerAlphaZeroNet
from mcts import AlphaZeroMCTS


def check_winner(obs_dict):
    if obs_dict is None:
        return -1
    if isinstance(obs_dict, dict):
        res = obs_dict.get("current", {}).get("result", -1)
        if isinstance(res, int) and res >= 0:
            return res
    return -1


def get_opponent_deck(opp_name: str):
    """各対戦相手の専用デッキを読み込む"""
    opp_deck_candidates = {
        "sampleAgent001": os.path.join(project_root, "dev", "sampleAgent001", "deck.csv"),
        "sampleAgent002": os.path.join(project_root, "dev", "sampleAgent002", "deck.csv"),
        "submit009": os.path.join(project_root, "dev", "submit009", "deck.csv"),
    }
    path = opp_deck_candidates.get(opp_name)
    if path and os.path.exists(path):
        return read_deck_csv(path)
    # fallback: our own deck
    return read_deck_csv(os.path.join(submit010_dir, "deck.csv"))


def run_eval(num_games: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(submit010_dir, "checkpoints", "latest_model.pt")
    deck_path = os.path.join(submit010_dir, "deck.csv")

    # Load submit010 model
    model = TransformerAlphaZeroNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    encoder = StateEncoder()
    mcts = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=100, device=device)

    our_deck = read_deck_csv(deck_path)

    opponents = [
        ("sampleAgent001", agent_sample001),
        ("sampleAgent002", agent_sample002),
        ("submit009",      agent_submit009),
    ]

    print("=" * 70)
    print(f"🎮 submit010 評価対局 (各{num_games}試合)")
    print(f"   モデル: {checkpoint_path}")
    print(f"   デバイス: {device}")
    print("=" * 70)

    all_results = {}

    for opp_name, opp_agent in opponents:
        wins = losses = draws = errors = 0
        opp_deck = get_opponent_deck(opp_name)

        print(f"\n🆚 vs {opp_name} ({num_games}試合)")
        print("-" * 50)

        for game_idx in range(num_games):
            obs_dict, _ = battle_start(our_deck, opp_deck)
            turn_count = 0
            winner = -1

            try:
                while obs_dict is not None and turn_count < 150:
                    turn_count += 1

                    w = check_winner(obs_dict)
                    if w >= 0:
                        winner = w
                        break

                    obs = to_observation_class(obs_dict)

                    # Determine current player
                    current_player = 0
                    if isinstance(obs_dict, dict) and "current" in obs_dict:
                        cur = obs_dict["current"]
                        if isinstance(cur, dict):
                            current_player = cur.get("player", 0)

                    if obs.select is None or not obs.select.option:
                        action = [0]
                    elif current_player == 0:
                        # submit010 acts
                        action_list, _, _ = mcts.get_action_distribution(obs, our_deck, opp_deck)
                        action = action_list
                    else:
                        # opponent acts
                        if opp_agent is not None and callable(opp_agent):
                            try:
                                action = opp_agent(obs_dict)
                            except Exception:
                                num_opts = len(obs.select.option)
                                min_cnt = max(1, getattr(obs.select, 'minCount', 1))
                                action = list(range(min(min_cnt, num_opts)))
                        else:
                            num_opts = len(obs.select.option)
                            min_cnt = max(1, getattr(obs.select, 'minCount', 1))
                            action = list(range(min(min_cnt, num_opts)))

                    # 安全クランプ: 選択肢数を超えたインデックスを修正
                    if obs.select and obs.select.option:
                        num_opts = len(obs.select.option)
                        action = [max(0, min(int(a), num_opts - 1)) for a in action]
                        if not action:
                            action = [0]

                    obs_dict = battle_select(action)

                    w = check_winner(obs_dict)
                    if w >= 0:
                        winner = w
                        break

            except Exception as e:
                errors += 1
                import traceback
                if errors <= 2:  # 最初の2エラーだけ詳細表示
                    print(f"  Game {game_idx+1:2d}: ❌ ERROR")
                    traceback.print_exc()
                else:
                    print(f"  Game {game_idx+1:2d}: ❌ ERROR ({type(e).__name__}: {e})")
                continue

            if winner == 0:
                wins += 1
                result_str = "✅ WIN"
            elif winner == 1:
                losses += 1
                result_str = "❌ LOSE"
            else:
                draws += 1
                result_str = "⬜ DRAW"

            print(f"  Game {game_idx+1:2d}: {result_str}  (turn={turn_count})")

        total = wins + losses + draws
        winrate = wins / total * 100 if total > 0 else 0
        print(f"\n  📊 結果 vs {opp_name}:")
        print(f"     WIN={wins}  LOSE={losses}  DRAW={draws}  ERROR={errors}")
        print(f"     勝率: {winrate:.1f}%  (有効対局: {total})")

        all_results[opp_name] = {
            "wins": wins, "losses": losses, "draws": draws,
            "errors": errors, "winrate": winrate
        }

    print("\n" + "=" * 70)
    print("🏆 総合評価結果")
    print("=" * 70)
    total_wins = total_losses = total_draws = 0
    for opp_name, r in all_results.items():
        print(f"  vs {opp_name:20s}: WIN={r['wins']:2d} LOSE={r['losses']:2d} DRAW={r['draws']:2d}  勝率={r['winrate']:.1f}%")
        total_wins += r["wins"]
        total_losses += r["losses"]
        total_draws += r["draws"]

    total_valid = total_wins + total_losses + total_draws
    overall_winrate = total_wins / total_valid * 100 if total_valid > 0 else 0
    print("-" * 70)
    print(f"  {'総合':24s}: WIN={total_wins:2d} LOSE={total_losses:2d} DRAW={total_draws:2d}  勝率={overall_winrate:.1f}%")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_eval(num_games=20)
