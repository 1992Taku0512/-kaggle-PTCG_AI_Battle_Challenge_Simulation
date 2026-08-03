import os
import sys
import time
import random
import torch
import torch.nn as nn
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")

if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType
from dev.submit003.state_encoder import StateEncoder
from dev.submit003.model import AlphaZeroNet
from dev.submit003.mcts import AlphaZeroMCTS, extract_option_type_val
from dev.submit003.main import read_deck_csv
from notify_line import send_line_notification


def load_deck_pool() -> list[tuple[str, list[int]]]:
    """Load all 10 decks from dev/deck_pool for diverse training experiences."""
    deck_pool_dir = os.path.join(project_root, "dev", "deck_pool")
    deck_files = [
        "deck_grass.csv", "deck_fire.csv", "deck_water.csv", "deck_lightning.csv",
        "deck_psychic.csv", "deck_fighting.csv", "deck_team_rocket.csv",
        "deck_ex_core.csv", "deck_mixed.csv", "deck_standard.csv"
    ]
    pool = []
    for df in deck_files:
        fp = os.path.join(deck_pool_dir, df)
        if os.path.exists(fp):
            with open(fp, "r") as f:
                lines = f.read().strip().split("\n")
            d = [int(line.strip()) for line in lines if line.strip()][:60]
            pool.append((df, d))
    if not pool:
        pool.append(("default_deck", read_deck_csv()))
    return pool


def extract_active_player(obs_dict: dict) -> int:
    """Extract active player index reliably from obs_dict."""
    if isinstance(obs_dict, dict):
        if "player" in obs_dict:
            return obs_dict["player"]
        if "current" in obs_dict and isinstance(obs_dict["current"], dict):
            return obs_dict["current"].get("yourIndex", 0)
    return 0


def check_match_finish(obs_dict: dict) -> tuple[bool, int]:
    """Check match completion and winner player index."""
    if isinstance(obs_dict, dict):
        if obs_dict.get("is_finish"):
            return True, obs_dict.get("winner", -1)
        if "current" in obs_dict and isinstance(obs_dict["current"], dict):
            res = obs_dict["current"].get("result", -1)
            if res >= 0:
                return True, res
    return False, -1


def compute_step_action_reward(obs, chosen_action_index: int) -> float:
    """Compute step-level reward / penalty based on deck-specific strategic actions."""
    if obs is None or obs.select is None or not obs.select.option:
        return 0.0
    
    opts = obs.select.option
    if chosen_action_index >= len(opts):
        return 0.0

    chosen_opt = opts[chosen_action_index]
    opt_type = extract_option_type_val(chosen_opt)

    step_reward = 0.0

    # 1. Positive Action Rewards (正の行動報酬)
    if opt_type == OptionType.ATTACK or opt_type == 8:
        step_reward += 0.3  # Attack execution
    elif opt_type in (OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY, 2, 4, 6):
        step_reward += 0.2  # Energy attach / Bench placement / Evolution

    # 2. Negative Penalties (不必要な行動・パスの罰則)
    if opt_type == OptionType.END or opt_type == 14:
        # Check if attack or play was available
        has_attack_or_play = any(
            extract_option_type_val(o) in (OptionType.ATTACK, OptionType.PLAY, 8, 2)
            for o in opts
        )
        if has_attack_or_play:
            step_reward -= 0.5  # Penalize unnecessary turn pass!

    return step_reward


def train_self_play(
    num_episodes: int = 50000,
    num_simulations: int = 150,
    batch_size: int = 512,
    epochs_per_episode: int = 2,
    notify_line: bool = True
):
    print("=" * 80)
    print(f"🚀 Starting Extended AlphaZero DRL Self-Play GPU Training ({num_episodes:,} Episodes)")
    print(f"   MCTS Simulations per Turn: {num_simulations}")
    print(f"   Mini-batch Size: {batch_size}")
    print(f"   Victory Reward: +5.0 / Defeat Penalty: -5.0")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ PyTorch Training Device: {device}")

    encoder = StateEncoder()
    model = AlphaZeroNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    weights_path = os.path.join(current_dir, "model_weights.pt")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"✅ Loaded initial checkpoint weights from: {weights_path}")
        except Exception as e:
            print(f"⚠️ Could not load initial weights ({e}), starting clean.")

    mcts_engine = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=num_simulations, device=device)
    deck_pool = load_deck_pool()
    my_deck = read_deck_csv()

    experience_buffer = []
    max_buffer_size = 30000
    recent_wins = []
    
    start_time = time.time()
    last_line_notify_time = start_time
    line_notify_interval_sec = 3600  # Notify every 1 hour (3600 seconds)

    for episode in range(1, num_episodes + 1):
        opp_name, opp_deck = random.choice(deck_pool)
        
        # Alternate Player 0 / Player 1 role
        our_player_idx = 0 if (episode % 2 == 1) else 1
        d0 = my_deck if our_player_idx == 0 else opp_deck
        d1 = opp_deck if our_player_idx == 0 else my_deck

        obs_dict, _ = battle_start(d0, d1)
        turn_count = 0
        max_turns = 200
        episode_experiences = []
        winner = -1
        prev_prizes = {0: 6, 1: 6}

        while obs_dict is not None and turn_count < max_turns:
            turn_count += 1
            is_done, match_winner = check_match_finish(obs_dict)
            if is_done:
                winner = match_winner
                break

            current_player = extract_active_player(obs_dict)
            obs = to_observation_class(obs_dict)

            # Prize tracking for intermediate KO reward shaping
            curr_prizes = {0: 6, 1: 6}
            if hasattr(obs, "current") and hasattr(obs.current, "players"):
                for p_idx, p_obj in enumerate(obs.current.players):
                    if hasattr(p_obj, "prize") and p_obj.prize is not None:
                        curr_prizes[p_idx] = len(p_obj.prize)

            if obs.select is None:
                action = my_deck
            else:
                state_vec = encoder.encode(obs)
                action_list, _, policy_target = mcts_engine.get_action_distribution(obs)
                action = action_list

                chosen_opt_idx = action_list[0] if action_list else 0
                step_reward = compute_step_action_reward(obs, chosen_opt_idx)

                # KO / Prize Card intermediate rewards
                if current_player == our_player_idx:
                    enemy_idx = 1 - our_player_idx
                    if curr_prizes[enemy_idx] < prev_prizes[enemy_idx]:
                        step_reward += 1.5 * (prev_prizes[enemy_idx] - curr_prizes[enemy_idx])  # Enemy KO reward
                    if curr_prizes[our_player_idx] < prev_prizes[our_player_idx]:
                        step_reward -= 1.5 * (prev_prizes[our_player_idx] - curr_prizes[our_player_idx])  # Own KO penalty

                episode_experiences.append({
                    "state": state_vec,
                    "policy_target": policy_target,
                    "player": current_player,
                    "step_reward": step_reward
                })

            prev_prizes = curr_prizes

            try:
                obs_dict = battle_select(action)
            except Exception:
                break

        # Check final outcome
        if winner < 0:
            _, winner = check_match_finish(obs_dict)

        is_win = (winner == our_player_idx)
        recent_wins.append(1 if is_win else 0)
        if len(recent_wins) > 500:
            recent_wins.pop(0)

        # Assign final game outcome reward (+5.0 heavy victory reward / -5.0 defeat penalty)
        outcome_reward = 5.0 if is_win else -5.0
        for exp in episode_experiences:
            # Combine final outcome reward with step-level action shaping reward
            final_target_value = (outcome_reward if exp["player"] == our_player_idx else -outcome_reward) + exp["step_reward"]
            final_target_value = max(-10.0, min(10.0, final_target_value))  # Clamp
            experience_buffer.append((exp["state"], exp["policy_target"], final_target_value))
            if len(experience_buffer) > max_buffer_size:
                experience_buffer.pop(0)

        battle_finish()

        # Stochastic Gradient Descent Training
        last_loss = 0.0
        if len(experience_buffer) >= batch_size:
            model.train()
            sample_size = min(batch_size * 2, len(experience_buffer))
            batch_indices = np.random.choice(len(experience_buffer), size=sample_size, replace=False)
            sampled_exps = [experience_buffer[idx] for idx in batch_indices]

            states_batch = torch.stack([exp[0] for exp in sampled_exps]).to(device)
            policies_batch = torch.tensor(np.array([exp[1] for exp in sampled_exps]), dtype=torch.float32, device=device)
            values_batch = torch.tensor([[exp[2]] for exp in sampled_exps], dtype=torch.float32, device=device)

            for epoch in range(epochs_per_episode):
                optimizer.zero_grad()
                pred_policy_logits, pred_values = model(states_batch)

                loss_value = torch.mean((pred_values - values_batch) ** 2)
                log_policy_probs = torch.log_softmax(pred_policy_logits, dim=-1)
                loss_policy = -torch.mean(torch.sum(policies_batch * log_policy_probs, dim=-1))
                loss_total = loss_value + loss_policy

                loss_total.backward()
                optimizer.step()
                last_loss = loss_total.item()

            model.eval()

        # Periodic Checkpoints & Console Progress
        if episode % 100 == 0 or episode == num_episodes:
            torch.save(model.state_dict(), weights_path)
            recent_winrate = (sum(recent_wins) / len(recent_wins)) * 100.0 if recent_wins else 0.0
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"Episode {episode:6d}/{num_episodes} | Winrate (last 500): {recent_winrate:5.1f}% | Loss: {last_loss:.4f} | Buffer: {len(experience_buffer):5d} | Elapsed: {elapsed_min:5.1f}m")

        # Hourly LINE Progress Notification (1時間に1回の定期報告)
        curr_now = time.time()
        if (curr_now - last_line_notify_time >= line_notify_interval_sec) or (episode == num_episodes and notify_line):
            elapsed_hours = (curr_now - start_time) / 3600.0
            progress_pct = (episode / num_episodes) * 100.0
            rem_hours = (elapsed_hours / episode) * (num_episodes - episode) if episode > 0 else 0.0
            recent_winrate = (sum(recent_wins) / len(recent_wins)) * 100.0 if recent_wins else 0.0

            hourly_msg = (
                f"⏰ 【PTCG AI 特訓 1時間進捗レポート】\n\n"
                f"・進捗: {episode:,} / {num_episodes:,} エピソード ({progress_pct:.1f}%)\n"
                f"・経過時間: {elapsed_hours:.1f} 時間 (残り推定: {rem_hours:.1f} 時間)\n"
                f"・直近500試合勝率: {recent_winrate:.1f}%\n"
                f"・最新 Loss: {last_loss:.4f}\n"
                f"・MCTS 探索数: {num_simulations} 回 / 手\n"
                f"・Replay Buffer: {len(experience_buffer):,} 局面\n\n"
                f"特訓は正常に進行中です！"
            )
            print("\n" + "=" * 60)
            print(hourly_msg)
            print("=" * 60 + "\n")
            if notify_line:
                send_line_notification(hourly_msg)
            last_line_notify_time = curr_now

    torch.save(model.state_dict(), weights_path)
    total_min = (time.time() - start_time) / 60.0
    print(f"\n🎉 Completed {num_episodes:,} Self-Play Training Episodes in {total_min:.1f} minutes!")
    return weights_path


if __name__ == "__main__":
    train_self_play(num_episodes=50000, num_simulations=150, batch_size=512, epochs_per_episode=2, notify_line=True)
