import os
import sys
import time
import subprocess
import re
import zipfile

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dev.submit004.train_selfplay import train_self_play
from notify_line import send_line_notification


def strip_ansi(text: str) -> str:
    """Strips ANSI control/escape characters from string."""
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def run_eval_subprocess(agent1_dir: str, agent2_dir: str, num_games: int = 20) -> dict:
    """Runs eval_local.py in an isolated subprocess."""
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
    """Package dev/submit004/submission.zip including cg module."""
    submit_dir = os.path.join(project_root, "dev", "submit004")
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

    print(f"📦 Built Final Submission Package: {zip_path} ({os.path.getsize(zip_path):,} bytes)")


def main():
    print("=" * 80)
    print("🚀 dev/submit004: 7-Hour Ultimate AlphaZero DRL Full Pipeline (50,000 Episodes + Reward Shaping)")
    print("=" * 80)

    start_train_time = time.time()
    
    # 1. 50,000 Episodes Training in dev/submit004
    train_self_play(
        num_episodes=50000,
        num_simulations=150,
        batch_size=512,
        epochs_per_episode=2,
        notify_line=True
    )
    
    train_duration_hours = (time.time() - start_train_time) / 3600.0
    print(f"\n✅ 50,000 Episodes GPU Self-Play Training Completed in {train_duration_hours:.2f} Hours!")

    # 2. Benchmark Evaluation vs 3 Agents (20 games each)
    print("\nPhase 2: Running Final Benchmark Evaluation for dev/submit004 (20 Games Each vs 3 Target Agents)...")
    opponents = [
        ("Kaggle Sample", "data/sample_submission/sample_submission"),
        ("submit001", "dev/submit001"),
        ("submit002", "dev/submit002"),
    ]

    results = []
    for name, opp_dir in opponents:
        print(f"\nEvaluating dev/submit004 vs {name} ({opp_dir}) for 20 games...")
        eval_res = run_eval_subprocess("dev/submit004", opp_dir, num_games=20)
        results.append((name, eval_res))
        print(f"-> {name} Result: {eval_res['a1_wins']}/20 Wins ({eval_res['winrate']:.1f}%) | Avg Turns: {eval_res['avg_turns']:.1f}")

    # 3. Zip Packaging
    build_submission_zip()

    # 4. Final LINE Notification
    msg_lines = [
        "🏆 【PTCG AI Battle】dev/submit004 7時間50,000エピソード完走 ＆ 最終評価レポート！",
        "",
        f"・モデル: dev/submit004 (submit003ベース報酬強化版)",
        f"・GPU学習時間: {train_duration_hours:.2f} 時間 (50,000 試合)",
        "・MCTS 探索数: 150 回 / 手",
        "・学習モデル: dev/submit004/model_weights.pt",
        "",
        "📊 【最終ベンチマーク勝率】"
    ]

    for name, res in results:
        msg_lines.append(f"・vs {name}: {res['a1_wins']}/20 勝 ({res['winrate']:.1f}%) [平均 {res['avg_turns']:.1f}T]")

    msg_lines.append("")
    msg_lines.append("📦 最新の提出用ZIP dev/submit004/submission.zip (cg同梱版) のパッケージングも完了いたしました！")
    final_msg = "\n".join(msg_lines)

    print("\n" + "=" * 80)
    print(final_msg)
    print("=" * 80)

    send_line_notification(final_msg)
    print("✅ Final LINE Notification Sent Successfully!")


if __name__ == "__main__":
    main()
