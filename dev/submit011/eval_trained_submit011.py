"""
submit011 訓練完了後モデル 総合評価スクリプト
10,000エピソード完了後の submit011 新モデルを用いて、各ベンチマーク相手および submit009 旧モデルに対する勝率を定量評価
"""
import os
import sys
import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit011_dir = os.path.abspath(os.path.dirname(__file__))
submit009_dir = os.path.join(project_root, "dev", "submit009")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit011_dir not in sys.path:
    sys.path.insert(0, submit011_dir)

sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)


def read_deck_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return [int(l.strip()) for l in lines if l.strip() and not l.startswith("#")][:60]


from dev.common.opponent_provider import (
    agent_sample001, agent_sample002, agent_submit009
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


def sanitize_action(obs, raw_action):
    if obs.select is None or not obs.select.option:
        return [0]
    num_opts = len(obs.select.option)
    min_cnt = max(1, getattr(obs.select, 'minCount', 1))
    max_cnt = getattr(obs.select, 'maxCount', min_cnt)

    if isinstance(raw_action, list):
        valid = [int(a) for a in raw_action if 0 <= int(a) < num_opts]
        if len(valid) >= min_cnt:
            return valid[:max_cnt]
    elif isinstance(raw_action, (int, np.integer)):
        if 0 <= raw_action < num_opts:
            return [int(raw_action)]

    return list(range(min(min_cnt, num_opts)))


def run_evaluation(num_games_per_opp: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    submit011_weights = os.path.join(submit011_dir, "model_weights.pt")

    print("=" * 70)
    print("🏆 submit011 (10,000ep 訓練完了モデル) ローカル総合評価")
    print(f"   モデル: {submit011_weights}")
    print(f"   デバイス: {device}")
    print("=" * 70)

    # 1. Load trained submit011 model
    model011 = TransformerAlphaZeroNet().to(device)
    ckpt011 = torch.load(submit011_weights, map_location=device, weights_only=False)
    state_dict011 = ckpt011.get("model_state_dict", ckpt011)
    model011.load_state_dict(state_dict011)
    model011.eval()

    encoder011 = StateEncoder()
    mcts011 = AlphaZeroMCTS(model=model011, encoder=encoder011, num_simulations=100, device=device)
    deck011 = read_deck_csv(os.path.join(submit011_dir, "deck.csv"))

    # 2. Load old submit009 model for direct head-to-head match
    model009 = TransformerAlphaZeroNet().to(device)
    ckpt009 = torch.load(os.path.join(submit009_dir, "model_weights.pt"), map_location=device, weights_only=False)
    state_dict009 = ckpt009.get("model_state_dict", ckpt009)
    model009.load_state_dict(state_dict009)
    model009.eval()

    mcts009 = AlphaZeroMCTS(model=model009, encoder=encoder011, num_simulations=100, device=device)
    deck009 = read_deck_csv(os.path.join(submit009_dir, "deck.csv"))

    opponents = [
        ("sampleAgent001", agent_sample001, os.path.join(project_root, "dev", "sampleAgent001", "deck.csv")),
        ("sampleAgent002", agent_sample002, os.path.join(project_root, "dev", "sampleAgent002", "deck.csv")),
        ("submit009 (旧王者)", agent_submit009, os.path.join(submit009_dir, "deck.csv")),
    ]

    all_results = {}

    for opp_name, opp_agent, opp_deck_path in opponents:
        opp_deck = read_deck_csv(opp_deck_path) if os.path.exists(opp_deck_path) else deck011
        wins = losses = draws = errors = 0

        print(f"\n🆚 vs {opp_name} ({num_games_per_opp}対局)")
        print("-" * 50)

        for game_idx in range(num_games_per_opp):
            obs_dict, _ = battle_start(deck011, opp_deck)
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
                    current_player = 0
                    if isinstance(obs_dict, dict) and "current" in obs_dict:
                        cur = obs_dict["current"]
                        if isinstance(cur, dict):
                            current_player = cur.get("player", 0)

                    if current_player == 0:
                        raw_action, _, _ = mcts011.get_action_distribution(obs, deck011, opp_deck)
                        action = sanitize_action(obs, raw_action)
                    else:
                        if opp_agent is not None and callable(opp_agent):
                            try:
                                raw_action = opp_agent(obs_dict)
                            except Exception:
                                raw_action = [0]
                        else:
                            raw_action = [0]
                        action = sanitize_action(obs, raw_action)

                    obs_dict = battle_select(action)
                    w = check_winner(obs_dict)
                    if w >= 0:
                        winner = w
                        break

            except Exception as e:
                errors += 1
                continue

            if winner == 0:
                wins += 1
                res_str = "✅ WIN"
            elif winner == 1:
                losses += 1
                res_str = "❌ LOSE"
            else:
                draws += 1
                res_str = "⬜ DRAW"

            print(f"  Game {game_idx+1:2d}: {res_str}  (turn={turn_count})")

        total = wins + losses + draws
        winrate = wins / total * 100 if total > 0 else 0
        print(f"  📊 結果 vs {opp_name}: WIN={wins} LOSE={losses} DRAW={draws} ERR={errors}  勝率: {winrate:.1f}%")
        all_results[opp_name] = {"wins": wins, "losses": losses, "draws": draws, "errors": errors, "winrate": winrate}

    # 3. Direct Head-to-Head: submit011 Trained Model vs submit009 Old Model
    print(f"\n==================================================")
    print(f"⚔️ 直接対決 (完全対決): 【submit011 訓練済モデル】 vs 【submit009 旧モデル】 ({num_games_per_opp}対局)")
    print(f"==================================================")

    new_wins = old_wins = draws = errors = 0
    for game_idx in range(num_games_per_opp):
        obs_dict, _ = battle_start(deck011, deck009)
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
                current_player = 0
                if isinstance(obs_dict, dict) and "current" in obs_dict:
                    cur = obs_dict["current"]
                    if isinstance(cur, dict):
                        current_player = cur.get("player", 0)

                if current_player == 0:
                    raw_action, _, _ = mcts011.get_action_distribution(obs, deck011, deck009)
                else:
                    raw_action, _, _ = mcts009.get_action_distribution(obs, deck009, deck011)

                action = sanitize_action(obs, raw_action)
                obs_dict = battle_select(action)

                w = check_winner(obs_dict)
                if w >= 0:
                    winner = w
                    break

        except Exception as e:
            errors += 1
            continue

        if winner == 0:
            new_wins += 1
            res = "✅ submit011 WIN"
        elif winner == 1:
            old_wins += 1
            res = "❌ submit009 WIN"
        else:
            draws += 1
            res = "⬜ DRAW"

        print(f"  Game {game_idx+1:2d}: {res} (turn={turn_count})")

    total = new_wins + old_wins + draws
    h2h_winrate = (new_wins / total * 100) if total > 0 else 0
    print(f"\n🏆 直接対決結果: submit011 勝率 {h2h_winrate:.1f}%  (WIN={new_wins}, LOSE={old_wins}, DRAW={draws})")

    print("\n" + "=" * 70)
    print("🏆 最終評価結果サマリー (submit011 訓練完了モデル)")
    print("=" * 70)
    for opp_name, r in all_results.items():
        print(f"  vs {opp_name:25s}: 勝率 {r['winrate']:5.1f}%  (WIN={r['wins']:2d}, LOSE={r['losses']:2d}, DRAW={r['draws']:2d})")
    print(f"  vs {'submit009 旧モデル (直接対決)':25s}: 勝率 {h2h_winrate:5.1f}%  (WIN={new_wins:2d}, LOSE={old_wins:2d}, DRAW={draws:2d})")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation(num_games_per_opp=15)
