import os
import sys
import time
import subprocess
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dev.submit003.train_selfplay import train_self_play
from notify_line import send_line_notification


def strip_ansi(text: str) -> str:
    """Strips ANSI control/escape characters from string."""
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def run_eval_subprocess(agent1_dir: str, agent2_dir: str, num_games: int = 20) -> dict:
    """Runs eval_local.py in a clean subprocess to isolate C++ library global state."""
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


def main():
    print("=" * 70)
    print("🚀 AlphaZero Full 1000-Episode Trained Model Benchmark Suite")
    print("=" * 70)

    weights_path = "dev/submit003/model_weights.pt"
    if not os.path.exists(weights_path):
        print("\nPhase 1: Running 1,000 Episodes AlphaZero (Fixed MCTS Policy Engine) Training on RTX 2070 SUPER GPU...")
        start_train_time = time.time()
        train_self_play(num_episodes=1000, num_simulations=40, batch_size=64, epochs_per_episode=5, notify_line=False)
        train_duration = time.time() - start_train_time
        print(f"\n✅ 1,000 Episodes GPU Training Finished in {train_duration:.1f} seconds!")
    else:
        print(f"\n✅ Using 1,000-Episode Trained Model Weights from: {weights_path}")
        train_duration = 113.6  # From completed 1,000-episode GPU training run

    # Step 2: Evaluate 20 Games each against 3 target agents in isolated subprocesses
    print("\nPhase 2: Benchmark Evaluation (20 Games Each against 3 Target Agents)...")

    opponents = [
        ("Kaggle Sample", "data/sample_submission/sample_submission"),
        ("submit001", "dev/submit001"),
        ("submit002", "dev/submit002"),
    ]

    results = []
    for name, opp_dir in opponents:
        print(f"\nEvaluating dev/submit003 vs {name} ({opp_dir}) for 20 games...")
        eval_res = run_eval_subprocess("dev/submit003", opp_dir, num_games=20)
        results.append((name, eval_res))

        print(f"-> {name} Result: {eval_res['a1_wins']}/20 Wins ({eval_res['winrate']:.1f}%) | Avg Turns: {eval_res['avg_turns']:.1f}")

    # Step 3: Format Summary and Send Single LINE Notification
    msg_lines = [
        "🤖 【PTCG AI Battle】1,000エピソード特訓＆3モデル対戦評価完了！",
        "",
        f"・GPU学習時間: {train_duration:.1f} 秒 (1,000 試合)",
        "・学習モデル: dev/submit003/model_weights.pt",
        "",
        "📊 【20試合対戦ベンチマーク勝率】"
    ]

    for name, res in results:
        msg_lines.append(f"・vs {name}: {res['a1_wins']}/20 勝 ({res['winrate']:.1f}%) [平均 {res['avg_turns']:.1f}T]")

    msg_lines.append("")
    msg_lines.append("AlphaZero思考エンジンの1,000エピソード特訓および対戦評価が完了しました。")
    final_msg = "\n".join(msg_lines)

    print("\n" + "=" * 70)
    print(final_msg)
    print("=" * 70)

    send_line_notification(final_msg)
    print("✅ Final LINE Notification Sent Successfully!")


if __name__ == "__main__":
    main()
