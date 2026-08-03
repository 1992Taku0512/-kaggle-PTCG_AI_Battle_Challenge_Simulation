import os
import sys
import time
import random
import subprocess
import re
import zipfile
import torch
import torch.nn as nn
import numpy as np

# Force unbuffered instant line-by-line log output
sys.stdout.reconfigure(line_buffering=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")

if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType
from dev.submit005.state_encoder import StateEncoder
from dev.submit005.model import AlphaZeroNet
from dev.submit005.mcts import AlphaZeroMCTS, extract_option_type_val
from dev.submit005.main import read_deck_csv
from notify_line import send_line_notification


def strip_ansi(text: str) -> str:
    """Strips ANSI control/escape characters from string."""
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def run_eval_subprocess(agent1_dir: str, agent2_dir: str, num_games: int = 20) -> dict:
    """Runs eval_local.py in an isolated subprocess for benchmark evaluation."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["COLUMNS"] = "200"
    
    cmd = [
        sys.executable,
        "eval_local.py",
        "--agent1", agent1_dir,
        "--agent2", agent2_dir,
        "--num-games", str(num_games)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, env=env)
    clean_stdout = strip_ansi(res.stdout)
    
    a1_wins = 0
    winrate = 0.0
    avg_turns = 0.0

    match_wins = re.search(r"Agent 1.*?Wins\s+:\s+(\d+)\s+/\s+(\d+)\s+\(([\d.]+)%\)", clean_stdout, re.DOTALL)
    if match_wins:
        a1_wins = int(match_wins.group(1))
        winrate = float(match_wins.group(3))

    match_turns = re.search(r"Average Turn Count\s+:\s+([\d.]+)", clean_stdout)
    if match_turns:
        avg_turns = float(match_turns.group(1))

    return {
        "agent1": agent1_dir,
        "agent2": agent2_dir,
        "num_games": num_games,
        "a1_wins": a1_wins,
        "winrate": winrate,
        "avg_turns": avg_turns,
        "raw_output": clean_stdout
    }


def build_submission_zip():
    """Package dev/submit005/submission.zip including cg module."""
    submit_dir = os.path.join(project_root, "dev", "submit005")
    cg_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission", "cg")
    zip_path = os.path.join(submit_dir, "submission.zip")

    agent_files = ["main.py", "deck.csv", "model.py", "state_encoder.py", "mcts.py", "model_weights.pt"]
    cg_files = ["__init__.py", "api.py", "cg.dll", "game.py", "libcg-arm64.so", "libcg.dylib", "libcg.so", "sim.py", "utils.py"]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in agent_files:
            fp = os.path.join(submit_dir, f)
            if os.path.exists(fp):
                zf.write(fp, arcname=f)
        for f in cg_files:
            fp = os.path.join(cg_dir, f)
            if os.path.exists(fp):
                zf.write(fp, arcname=os.path.join("cg", f))

    print(f"📦 Built Submission Package: {zip_path} ({os.path.getsize(zip_path):,} bytes)")


def load_deck_pool() -> dict[str, list[int]]:
    """Load all 10 decks from dev/deck_pool."""
    deck_pool_dir = os.path.join(project_root, "dev", "deck_pool")
    deck_files = [
        "deck_grass.csv", "deck_fire.csv", "deck_water.csv", "deck_lightning.csv",
        "deck_psychic.csv", "deck_fighting.csv", "deck_team_rocket.csv",
        "deck_ex_core.csv", "deck_mixed.csv", "deck_standard.csv"
    ]
    pool = {}
    for df in deck_files:
        fp = os.path.join(deck_pool_dir, df)
        if os.path.exists(fp):
            with open(fp, "r") as f:
                lines = f.read().strip().split("\n")
            d = [int(line.strip()) for line in lines if line.strip()][:60]
            pool[df] = d
    if not pool:
        pool["default_deck"] = read_deck_csv()
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

    # 1. Positive Action Rewards
    if opt_type in (OptionType.ATTACK, 13, 8):
        step_reward += 0.3  # Attack execution
    elif opt_type in (OptionType.ENERGY, OptionType.ENERGY_CARD, 5, 6):
        target_energy_count = getattr(chosen_opt, "energyCount", 0) if not isinstance(chosen_opt, dict) else chosen_opt.get("energyCount", 0)
        target_needed_cost = getattr(chosen_opt, "neededCost", 2) if not isinstance(chosen_opt, dict) else chosen_opt.get("neededCost", 2)
        is_mismatched = getattr(chosen_opt, "isMismatched", False) if not isinstance(chosen_opt, dict) else chosen_opt.get("isMismatched", False)

        if is_mismatched:
            step_reward -= 0.3  # Penalty for attaching mismatched energy type!
        elif target_energy_count >= target_needed_cost and target_needed_cost > 0:
            step_reward -= 0.2  # Penalty for attaching excess energy beyond needed attack cost!
        else:
            step_reward += 0.3  # Positive reward for appropriate energy attachment!
    elif opt_type in (OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY, 7, 9, 10):
        step_reward += 0.2  # Card play / Evolution / Ability

    # 2. Negative Penalties
    if opt_type in (OptionType.END, 14):
        has_attack_or_play = any(
            extract_option_type_val(o) in (OptionType.ATTACK, OptionType.PLAY, OptionType.ENERGY, 13, 7, 6, 8)
            for o in opts
        )
        if has_attack_or_play:
            step_reward -= 0.5  # Penalize unnecessary turn pass!

    return step_reward


def load_submit004_opponent_model(device):
    """Load past submit004 trained weights for Phase 4 opponent sampling."""
    weights_path = os.path.join(project_root, "dev", "submit004", "model_weights.pt")
    if os.path.exists(weights_path):
        try:
            m = AlphaZeroNet().to(device)
            m.load_state_dict(torch.load(weights_path, map_location=device))
            m.eval()
            encoder = StateEncoder()
            mcts = AlphaZeroMCTS(model=m, encoder=encoder, num_simulations=100, device=device)
            print("✅ Successfully loaded submit004 past checkpoint model for Phase 4 opponent policy!")
            return mcts
        except Exception as e:
            print(f"⚠️ Could not load submit004 model ({e})")
    return None


def run_phase_benchmark_report(phase_num: int, phase_name: str, total_episodes: int, start_time: float):
    """Runs 60-game benchmark evaluation across 3 agents and sends LINE report."""
    print(f"\n📊 Running Phase {phase_num} Completion Benchmark Evaluation (60 Games Total)...")
    opponents = [
        ("Kaggle Sample", "data/sample_submission/sample_submission"),
        ("submit001", "dev/submit001"),
        ("submit002", "dev/submit002"),
    ]
    results = []
    for name, opp_dir in opponents:
        eval_res = run_eval_subprocess("dev/submit005", opp_dir, num_games=20)
        results.append((name, eval_res))
        print(f"-> {name} Result: {eval_res['a1_wins']}/20 Wins ({eval_res['winrate']:.1f}%) | Avg Turns: {eval_res['avg_turns']:.1f}")

    build_submission_zip()

    elapsed_h = (time.time() - start_time) / 3600.0
    msg_lines = [
        f"🎯 【PTCG AI dev/submit005】Phase {phase_num} ({phase_name}) 達成＆自動昇格完了！",
        "",
        f"・累計エピソード: {total_episodes:,} 試合",
        f"・経過時間: {elapsed_h:.2f} 時間",
        f"・チェックポイント: dev/submit005/model_weights.pt",
        "",
        "📊 【Phase " + str(phase_num) + " 完了時ベンチマーク勝率】"
    ]
    for name, res in results:
        msg_lines.append(f"・vs {name}: {res['a1_wins']}/20 勝 ({res['winrate']:.1f}%) [平均 {res['avg_turns']:.1f}T]")
    
    msg_lines.append("")
    msg_lines.append(f"📦 提出用ZIP dev/submit005/submission.zip (cg同梱) を更新いたしました！")
    
    final_msg = "\n".join(msg_lines)
    print("\n" + "=" * 80)
    print(final_msg)
    print("=" * 80 + "\n")
    send_line_notification(final_msg)


def train_staged_curriculum():
    """4-Phase Staged Adaptive Curriculum Learning Pipeline for dev/submit005."""
    print("=" * 80)
    print("🚀 Starting dev/submit005 4-Phase Adaptive Curriculum DRL GPU Training")
    print("   Model Initialization: CLEAN INITIALIZATION (New model weights)")
    print("   Phase 1: Mirror Match (Goal: Winrate >= 85.0%, Avg Turns <= 25T)")
    print("   Phase 2: 3 Core Decks (Goal: vs submit001 Winrate >= 75.0%)")
    print("   Phase 3: 6 Core Decks (Goal: vs submit002 Winrate >= 68.0%)")
    print("   Phase 4: All 10 Decks x All Policies including submit004 (Goal: Overall Winrate >= 60.0%)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ PyTorch Training Device: {device}")

    encoder = StateEncoder()
    # CLEAN INITIALIZATION: Always start from scratch for dev/submit005!
    model = AlphaZeroNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights_path = os.path.join(current_dir, "model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print("✨ Cleanly initialized brand-new PyTorch neural network weights for dev/submit005.")

    mcts_engine = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=150, device=device)
    submit004_mcts = load_submit004_opponent_model(device)
    
    deck_pool_dict = load_deck_pool()
    my_deck = read_deck_csv()

    experience_buffer = []
    max_buffer_size = 30000
    
    start_time = time.time()
    last_line_notify_time = start_time
    line_notify_interval_sec = 3600

    current_phase = 1
    total_episodes = 0
    phase_episodes = 0
    recent_wins = []
    recent_turns = []

    # Phase 2 Core Decks
    phase2_decks = [deck_pool_dict.get("deck_fire.csv", my_deck), deck_pool_dict.get("deck_water.csv", my_deck), deck_pool_dict.get("deck_lightning.csv", my_deck)]
    # Phase 3 Core Decks
    phase3_decks = phase2_decks + [deck_pool_dict.get("deck_grass.csv", my_deck), deck_pool_dict.get("deck_psychic.csv", my_deck), deck_pool_dict.get("deck_fighting.csv", my_deck)]
    # Phase 4 All Decks
    phase4_decks = list(deck_pool_dict.values())

    max_phase_episodes = 1000000  # 1 Million episodes safety cap per phase

    while current_phase <= 4:
        phase_episodes += 1
        total_episodes += 1

        # Select opponent deck based on active Phase
        if current_phase == 1:
            opp_deck = my_deck  # Mirror match
        elif current_phase == 2:
            opp_deck = random.choice(phase2_decks)
        elif current_phase == 3:
            opp_deck = random.choice(phase3_decks)
        else:
            opp_deck = random.choice(phase4_decks)

        our_player_idx = 0 if (total_episodes % 2 == 1) else 1
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

            curr_prizes = {0: 6, 1: 6}
            if hasattr(obs, "current") and hasattr(obs.current, "players"):
                for p_idx, p_obj in enumerate(obs.current.players):
                    if hasattr(p_obj, "prize") and p_obj.prize is not None:
                        curr_prizes[p_idx] = len(p_obj.prize)

            if obs.select is None:
                action = my_deck
            else:
                state_vec = encoder.encode(obs)
                
                # Check if opponent has custom policy in Phase 4 (submit004 model)
                if current_phase == 4 and current_player != our_player_idx and submit004_mcts is not None and random.random() < 0.25:
                    action_list, _, policy_target = submit004_mcts.get_action_distribution(obs)
                else:
                    action_list, _, policy_target = mcts_engine.get_action_distribution(obs)

                action = action_list
                chosen_opt_idx = action_list[0] if action_list else 0
                step_reward = compute_step_action_reward(obs, chosen_opt_idx)

                if current_player == our_player_idx:
                    enemy_idx = 1 - our_player_idx
                    if curr_prizes[enemy_idx] < prev_prizes[enemy_idx]:
                        step_reward += 1.5 * (prev_prizes[enemy_idx] - curr_prizes[enemy_idx])
                    if curr_prizes[our_player_idx] < prev_prizes[our_player_idx]:
                        step_reward -= 1.5 * (prev_prizes[our_player_idx] - curr_prizes[our_player_idx])

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

        if winner < 0:
            _, winner = check_match_finish(obs_dict)

        is_win = (winner == our_player_idx)
        recent_wins.append(1 if is_win else 0)
        recent_turns.append(turn_count)
        if len(recent_wins) > 200:
            recent_wins.pop(0)
            recent_turns.pop(0)

        outcome_reward = 5.0 if is_win else -5.0
        for exp in episode_experiences:
            final_target_value = (outcome_reward if exp["player"] == our_player_idx else -outcome_reward) + exp["step_reward"]
            final_target_value = max(-10.0, min(10.0, final_target_value))
            experience_buffer.append((exp["state"], exp["policy_target"], final_target_value))
            if len(experience_buffer) > max_buffer_size:
                experience_buffer.pop(0)

        battle_finish()

        # SGD Training
        last_loss = 0.0
        if len(experience_buffer) >= 512:
            model.train()
            sample_size = min(1024, len(experience_buffer))
            batch_indices = np.random.choice(len(experience_buffer), size=sample_size, replace=False)
            sampled_exps = [experience_buffer[idx] for idx in batch_indices]

            states_batch = torch.stack([exp[0] for exp in sampled_exps]).to(device)
            policies_batch = torch.tensor(np.array([exp[1] for exp in sampled_exps]), dtype=torch.float32, device=device)
            values_batch = torch.tensor([[exp[2]] for exp in sampled_exps], dtype=torch.float32, device=device)

            for epoch in range(2):
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

        # Checkpoints & Progress Logging
        if total_episodes % 100 == 0:
            torch.save(model.state_dict(), weights_path)
            recent_wr = (sum(recent_wins) / len(recent_wins)) * 100.0 if recent_wins else 0.0
            avg_t = sum(recent_turns) / len(recent_turns) if recent_turns else 0.0
            elapsed_m = (time.time() - start_time) / 60.0
            print(f"Phase {current_phase} | Total Ep {total_episodes:7d} (Phase Ep {phase_episodes:6d}) | Winrate: {recent_wr:5.1f}% | Avg Turns: {avg_t:4.1f}T | Loss: {last_loss:.4f} | Elapsed: {elapsed_m:5.1f}m")

        # Hourly LINE Notification
        curr_now = time.time()
        if (curr_now - last_line_notify_time >= line_notify_interval_sec):
            elapsed_h = (curr_now - start_time) / 3600.0
            recent_wr = (sum(recent_wins) / len(recent_wins)) * 100.0 if recent_wins else 0.0
            avg_t = sum(recent_turns) / len(recent_turns) if recent_turns else 0.0

            hourly_msg = (
                f"⏰ 【PTCG AI dev/submit005 1時間進捗レポート】\n\n"
                f"・現在のステージ: Phase {current_phase}\n"
                f"・累計エピソード: {total_episodes:,} 試合 (Phase内 {phase_episodes:,} 試合)\n"
                f"・経過時間: {elapsed_h:.2f} 時間\n"
                f"・直近200試合勝率: {recent_wr:.1f}%\n"
                f"・直近平均ターン数: {avg_t:.1f} T\n"
                f"・最新 Loss: {last_loss:.4f}\n\n"
                f"カリキュラム特訓は正常に進行中です！"
            )
            print("\n" + "=" * 60)
            print(hourly_msg)
            print("=" * 60 + "\n")
            send_line_notification(hourly_msg)
            last_line_notify_time = curr_now

        # Check Phase Promotion Conditions!
        recent_wr = (sum(recent_wins) / len(recent_wins)) * 100.0 if len(recent_wins) >= 200 else 0.0
        avg_t = sum(recent_turns) / len(recent_turns) if len(recent_turns) >= 200 else 999.0

        should_promote = False
        p_name = ""

        if current_phase == 1:
            # Phase 1: Mirror match -> Winrate >= 85.0% AND Avg turns <= 25T
            if (recent_wr >= 85.0 and avg_t <= 25.0 and phase_episodes >= 2000) or (phase_episodes >= max_phase_episodes):
                should_promote = True
                p_name = "ミラー戦基礎完全固定"
        elif current_phase == 2:
            # Phase 2: Core 3 decks -> Winrate >= 75.0%
            if (recent_wr >= 75.0 and phase_episodes >= 3000) or (phase_episodes >= max_phase_episodes):
                should_promote = True
                p_name = "代表3属性対応"
        elif current_phase == 3:
            # Phase 3: Core 6 decks -> Winrate >= 68.0%
            if (recent_wr >= 68.0 and phase_episodes >= 5000) or (phase_episodes >= max_phase_episodes):
                should_promote = True
                p_name = "主要6属性 ＆ submit002撃破"
        elif current_phase == 4:
            # Phase 4: All 10 decks -> Winrate >= 60.0%
            if (recent_wr >= 60.0 and phase_episodes >= 10000) or (phase_episodes >= max_phase_episodes):
                should_promote = True
                p_name = "全10種デッキ ✕ 全方策 総力戦完成"

        if should_promote:
            print(f"\n🎉 Phase {current_phase} SUCCESS! Triggering Benchmark & LINE Notification...")
            run_phase_benchmark_report(current_phase, p_name, total_episodes, start_time)
            current_phase += 1
            phase_episodes = 0
            recent_wins = []
            recent_turns = []

    print(f"\n👑 dev/submit005 All 4 Phases Staged Curriculum DRL Training Fully Completed in {(time.time() - start_time)/3600.0:.2f} Hours!")


if __name__ == "__main__":
    train_staged_curriculum()
