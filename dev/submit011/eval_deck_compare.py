"""
submit011 デッキ比較評価スクリプト
submit009モデルを用いて、旧35エネデッキ vs 新24エネ改良デッキの性能・勝率・挙動を比較評価
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
if submit009_dir not in sys.path:
    sys.path.insert(0, submit009_dir)

sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)


def read_deck_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return [int(l.strip()) for l in lines if l.strip() and not l.startswith("#")][:60]


from dev.common.opponent_provider import (
    agent_sample001, agent_sample002
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
    """Ensure action is within valid option bounds and respects minCount / maxCount."""
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


def evaluate_deck(model, mcts, deck_name: str, test_deck: list, num_games_per_opp: int = 15):
    opponents = [
        ("sampleAgent001", agent_sample001, os.path.join(project_root, "dev", "sampleAgent001", "deck.csv")),
        ("sampleAgent002", agent_sample002, os.path.join(project_root, "dev", "sampleAgent002", "deck.csv")),
    ]

    print(f"\n==================================================")
    print(f"📊 テスト対象: 【{deck_name}】 (各{num_games_per_opp}対局)")
    print(f"==================================================")

    results = {}

    for opp_name, opp_agent, opp_deck_path in opponents:
        opp_deck = read_deck_csv(opp_deck_path) if os.path.exists(opp_deck_path) else test_deck
        wins = losses = draws = errors = 0

        for game_idx in range(num_games_per_opp):
            obs_dict, _ = battle_start(test_deck, opp_deck)
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
                        raw_action, _, _ = mcts.get_action_distribution(obs, test_deck, opp_deck)
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
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        total = wins + losses + draws
        winrate = wins / total * 100 if total > 0 else 0
        print(f"  vs {opp_name:16s}: 勝率 {winrate:5.1f}%  (WIN={wins:2d}, LOSE={losses:2d}, DRAW={draws:2d}, ERR={errors:2d})")
        results[opp_name] = winrate

    return results


def run_head_to_head(mcts, new_deck: list, old_deck: list, num_games: int = 15):
    print(f"\n==================================================")
    print(f"⚔️ 直接対決: 【新24エネデッキ】 vs 【旧35エネデッキ】 ({num_games}対局)")
    print(f"==================================================")

    new_wins = old_wins = draws = errors = 0

    for game_idx in range(num_games):
        obs_dict, _ = battle_start(new_deck, old_deck)
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
                    raw_action, _, _ = mcts.get_action_distribution(obs, new_deck, old_deck)
                else:
                    raw_action, _, _ = mcts.get_action_distribution(obs, old_deck, new_deck)

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
            res = "新24エネ WIN"
        elif winner == 1:
            old_wins += 1
            res = "旧35エネ WIN"
        else:
            draws += 1
            res = "引き分け"

        print(f"  Game {game_idx+1:2d}: {res} (turn={turn_count})")

    total = new_wins + old_wins + draws
    winrate = (new_wins / total * 100) if total > 0 else 0
    print(f"\n🏆 直接対決結果: 新24エネ勝率 {winrate:.1f}%")
    print(f"   新24エネ勝数={new_wins}  旧35エネ勝数={old_wins}  引き分け={draws}  エラー={errors}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(submit009_dir, "model_weights.pt")

    model = TransformerAlphaZeroNet().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    encoder = StateEncoder()
    mcts = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=50, device=device)

    old_deck = read_deck_csv(os.path.join(submit009_dir, "deck.csv"))
    new_deck = read_deck_csv(os.path.join(submit011_dir, "deck.csv"))

    res_old = evaluate_deck(model, mcts, "旧 35エネデッキ (submit009)", old_deck, num_games_per_opp=15)
    res_new = evaluate_deck(model, mcts, "新 24エネデッキ (submit011)", new_deck, num_games_per_opp=15)
    run_head_to_head(mcts, new_deck, old_deck, num_games=15)


if __name__ == "__main__":
    main()
