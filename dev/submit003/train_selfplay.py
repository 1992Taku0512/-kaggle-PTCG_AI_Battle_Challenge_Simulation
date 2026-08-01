import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

sample_dir = os.path.abspath("data/sample_submission/sample_submission")
if os.path.exists(sample_dir) and sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish
from notify_line import send_line_notification

from state_encoder import StateEncoder
from model import AlphaZeroNet
from mcts import AlphaZeroMCTS
from dev.submit003.main import agent as submit003_agent, read_deck_csv


def load_deck_pool() -> List[List[int]]:
    """Loads all 60-card decks from dev/deck_pool/."""
    pool_dir = os.path.abspath(os.path.join(current_dir, "../deck_pool"))
    decks = []
    if os.path.exists(pool_dir):
        for fname in sorted(os.listdir(pool_dir)):
            if fname.endswith(".csv"):
                fpath = os.path.join(pool_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        lines = [int(line.strip()) for line in f.read().strip().split("\n") if line.strip()]
                    if len(lines) == 60:
                        decks.append(lines)
                except Exception:
                    pass
    if not decks:
        decks.append(read_deck_csv())
    return decks


def train_self_play(num_episodes: int = 20, num_simulations: int = 30, batch_size: int = 64, epochs_per_episode: int = 5, notify_line: bool = True):
    """Multi-Deck Multi-Opponent AlphaZero Training Pipeline on RTX 2070 SUPER GPU."""
    import random
    print("=" * 60)
    print("🚀 Starting AlphaZero Multi-Deck Self-Play Training Loop on GPU")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = StateEncoder()
    model = AlphaZeroNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    weights_path = os.path.join(current_dir, "model_weights.pt")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded existing model weights from: {weights_path}")
        except Exception as e:
            print(f"Could not load weights ({e}), initializing fresh model.")

    my_deck = read_deck_csv()
    deck_pool = load_deck_pool()
    print(f"Loaded {len(deck_pool)} distinct opponent 60-card decks in dev/deck_pool/")

    mcts_engine = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=num_simulations, device=device)
    max_buffer_size = 5000
    experience_buffer: List[Tuple[torch.Tensor, np.ndarray, float]] = []

    start_time = time.time()
    total_wins = 0

    for episode in range(1, num_episodes + 1):
        # Player 0 (Our Agent) is FIXED to submit003 deck
        # Player 1 (Opponent) is RANDOMLY SAMPLED from deck pool
        opp_deck = random.choice(deck_pool)
        
        # Alternate going first vs going second
        if episode % 2 == 1:
            d0, d1 = my_deck, opp_deck
            our_player_idx = 0
        else:
            d0, d1 = opp_deck, my_deck
            our_player_idx = 1
        
        if episode % 50 == 0 or episode == 1:
            print(f"--- Episode {episode}/{num_episodes} (Our Agent: submit003 deck | Opponent Deck Pool Sample) ---")
        
        obs_dict, _ = battle_start(d0, d1)
        episode_experiences = []
        turn_count = 0
        max_turns = 200

        while obs_dict is not None and turn_count < max_turns:
            turn_count += 1
            if isinstance(obs_dict, dict) and obs_dict.get("is_finish"):
                winner = obs_dict.get("winner", -1)
                if winner == our_player_idx:
                    total_wins += 1
                break

            current_player = obs_dict.get("player", 0)
            obs = to_observation_class(obs_dict)

            if obs.select is None:
                action = my_deck
            else:
                # 1. State Encoding
                state_vec = encoder.encode(obs)

                # 2. MCTS Search
                action_list, _, policy_target = mcts_engine.get_action_distribution(obs)
                action = action_list

                # Store trajectory (state_tensor, policy_target_vector, player_index)
                episode_experiences.append({
                    "state": state_vec,
                    "policy_target": policy_target,
                    "player": current_player
                })

            try:
                obs_dict = battle_select(action)
            except Exception:
                break

        # Calculate final outcome reward (+1 for winner, -1 for loser)
        winner = obs_dict.get("winner", 0) if isinstance(obs_dict, dict) else 0
        for exp in episode_experiences:
            z_reward = 1.0 if exp["player"] == winner else -1.0
            experience_buffer.append((exp["state"], exp["policy_target"], z_reward))
            if len(experience_buffer) > max_buffer_size:
                experience_buffer.pop(0)

        battle_finish()

        # 3. GPU Neural Network Training Step (Mini-Batch Stochastic Gradient Descent)
        if len(experience_buffer) >= batch_size:
            model.train()
            
            # Sample random mini-batch from sliding replay buffer
            sample_size = min(batch_size * 4, len(experience_buffer))
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

            model.eval()

            if episode % 50 == 0 or episode == num_episodes:
                torch.save(model.state_dict(), weights_path)
                print(f"✅ Model weights saved to: {weights_path} | Total Loss: {loss_total.item():.4f} (Val: {loss_value.item():.4f}, Pol: {loss_policy.item():.4f})")

    elapsed_time = time.time() - start_time
    winrate = (total_wins / num_episodes) * 100.0

    print("=" * 60)
    print(f"🎉 AlphaZero Self-Play Training Completed in {elapsed_time:.2f} seconds!")
    print(f"Total Episodes: {num_episodes} | Winrate: {winrate:.1f}%")
    print("=" * 60)

    # LINE Notification
    if notify_line:
        msg = (
            f"🤖 【PTCG AI Battle】AlphaZero DRL 学習完了通知！\n\n"
            f"・エピソード数: {num_episodes} 試合\n"
            f"・学習時間: {elapsed_time:.1f} 秒\n"
            f"・モデル勝率: {winrate:.1f}%\n"
            f"・保存ファイル: submit003/model_weights.pt\n\n"
            f"RTX 2070 SUPER（GPU）による深層強化学習と重み更新が完了しました。"
        )
        send_line_notification(msg)


if __name__ == "__main__":
    train_self_play(num_episodes=5, num_simulations=20, batch_size=16)
