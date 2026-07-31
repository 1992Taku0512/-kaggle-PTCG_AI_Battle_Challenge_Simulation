import argparse
import importlib.util
import os
import sys
from tqdm import tqdm

def load_agent(agent_dir: str):
    """Load agent function and deck.csv from a specified directory."""
    abs_dir = os.path.abspath(agent_dir)
    main_path = os.path.join(abs_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"main.py not found in {abs_dir}")
    
    # Load module dynamically
    clean_dir = abs_dir.replace('/', '_').replace('\\', '_')
    module_name = f"agent_module_{clean_dir}"

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    module = importlib.util.module_from_spec(spec)
    
    # Ensure sample_submission is in sys.path so cg can be imported
    sample_dir = os.path.abspath("data/sample_submission/sample_submission")
    if sample_dir not in sys.path:
        sys.path.insert(0, sample_dir)
        
    # Save cwd and switch temporarily for module loading if needed
    orig_cwd = os.getcwd()
    try:
        os.chdir(abs_dir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(orig_cwd)

        
    return module.agent, module.read_deck_csv, abs_dir

def run_single_match(agent0_info, agent1_info, max_turns: int = 1000):
    """Run a single battle between two agents.
    
    Returns:
        dict: Match result stats
    """
    agent0_fn, read_deck0, dir0 = agent0_info
    agent1_fn, read_deck1, dir1 = agent1_info
    
    # Read decks
    orig_cwd = os.getcwd()
    try:
        os.chdir(dir0)
        deck0 = read_deck0()
        os.chdir(dir1)
        deck1 = read_deck1()
    finally:
        os.chdir(orig_cwd)
        
    # Setup imports for cg
    sample_dir = os.path.abspath("data/sample_submission/sample_submission")
    if sample_dir not in sys.path:
        sys.path.insert(0, sample_dir)
    
    try:
        os.chdir(sample_dir)
        from cg.game import battle_start, battle_select, battle_finish
        
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            return {"winner": None, "turns": 0, "status": "init_failed"}
        
        turn_count = 0
        winner = None
        status = "finished"
        
        while obs is not None and turn_count < max_turns:
            turn_count += 1
            if isinstance(obs, dict) and obs.get("is_finish"):
                winner = obs.get("winner")
                break
                
            current_player = obs.get("player", 0)
            
            # Select active agent
            if current_player == 0 or current_player is None:
                current_agent = agent0_fn
                current_dir = dir0
            else:
                current_agent = agent1_fn
                current_dir = dir1
                
            try:
                os.chdir(current_dir)
                action = current_agent(obs)
            except Exception as e:
                # Agent code crashed
                winner = 1 if (current_player == 0 or current_player is None) else 0
                status = f"error_agent_{current_player}: {type(e).__name__}"
                break
            finally:
                os.chdir(sample_dir)
                
            try:
                obs = battle_select(action)
            except Exception as e:
                # Engine exception or invalid move
                status = f"engine_exception: {type(e).__name__}"
                break
                
        battle_finish()
        return {"winner": winner, "turns": turn_count, "status": status}
    finally:
        os.chdir(orig_cwd)

def evaluate(agent1_dir: str, agent2_dir: str, num_games: int, max_turns: int):
    print(f"=== PTCG AI Battle Local Evaluation ===")
    print(f"Agent 1: {agent1_dir}")
    print(f"Agent 2: {agent2_dir}")
    print(f"Total Games: {num_games} (First/Second swapped evenly)")
    print("-" * 50)
    
    agent1_info = load_agent(agent1_dir)
    agent2_info = load_agent(agent2_dir)
    
    stats = {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws_or_unfinished": 0,
        "agent1_first_wins": 0,
        "agent1_second_wins": 0,
        "agent2_first_wins": 0,
        "agent2_second_wins": 0,
        "total_turns": 0,
        "errors": 0,
        "error_reasons": {}
    }

    
    for i in tqdm(range(num_games), desc="Evaluating"):
        # Swap first/second player every match
        if i % 2 == 0:
            # Agent 1 is Player 0 (First), Agent 2 is Player 1 (Second)
            res = run_single_match(agent1_info, agent2_info, max_turns)
            winner = res["winner"]
            stats["total_turns"] += res["turns"]
            
            if winner == 0:
                stats["agent1_wins"] += 1
                stats["agent1_first_wins"] += 1
            elif winner == 1:
                stats["agent2_wins"] += 1
                stats["agent2_second_wins"] += 1
            else:
                stats["draws_or_unfinished"] += 1
        else:
            # Agent 2 is Player 0 (First), Agent 1 is Player 1 (Second)
            res = run_single_match(agent2_info, agent1_info, max_turns)
            winner = res["winner"]
            stats["total_turns"] += res["turns"]
            
            if winner == 0:
                stats["agent2_wins"] += 1
                stats["agent2_first_wins"] += 1
            elif winner == 1:
                stats["agent1_wins"] += 1
                stats["agent1_second_wins"] += 1
            else:
                stats["draws_or_unfinished"] += 1
                
        if "error" in res["status"] or "engine_exception" in res["status"]:
            stats["errors"] += 1
            reason = res["status"]
            stats["error_reasons"][reason] = stats["error_reasons"].get(reason, 0) + 1


    print("\n" + "=" * 50)
    print(" SUMMARY RESULTS")
    print("=" * 50)
    winrate1 = (stats["agent1_wins"] / num_games) * 100
    winrate2 = (stats["agent2_wins"] / num_games) * 100
    avg_turns = stats["total_turns"] / num_games
    
    print(f"Agent 1 ({agent1_dir}) Wins : {stats['agent1_wins']} / {num_games} ({winrate1:.1f}%)")
    print(f"  - First-player Wins  : {stats['agent1_first_wins']} / {num_games // 2}")
    print(f"  - Second-player Wins : {stats['agent1_second_wins']} / {num_games - num_games // 2}")
    print()
    print(f"Agent 2 ({agent2_dir}) Wins : {stats['agent2_wins']} / {num_games} ({winrate2:.1f}%)")
    print(f"  - First-player Wins  : {stats['agent2_first_wins']} / {num_games - num_games // 2}")
    print(f"  - Second-player Wins : {stats['agent2_second_wins']} / {num_games // 2}")
    print()
    print(f"Draws / Unfinished : {stats['draws_or_unfinished']}")
    print(f"Engine/Agent Errors : {stats['errors']}")
    print(f"Average Turn Count  : {avg_turns:.1f}")
    if stats["error_reasons"]:
        print("\nError Reasons breakdown:")
        for reason, count in stats["error_reasons"].items():
            print(f"  - {reason}: {count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PTCG AI Battle Agents locally.")
    parser.add_argument("--agent1", type=str, default="dev/submit001", help="Path to Agent 1 directory")
    parser.add_argument("--agent2", type=str, default="data/sample_submission/sample_submission", help="Path to Agent 2 directory")
    parser.add_argument("--num-games", type=int, default=20, help="Number of games to simulate")
    parser.add_argument("--max-turns", type=int, default=1000, help="Maximum turns per game")
    
    args = parser.parse_args()
    evaluate(args.agent1, args.agent2, args.num_games, args.max_turns)
