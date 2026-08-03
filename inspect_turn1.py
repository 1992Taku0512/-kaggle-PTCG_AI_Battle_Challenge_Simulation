import sys
import os

sample_dir = os.path.abspath("data/sample_submission/sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.game import battle_start, battle_select, battle_finish
from eval_local import load_agent

a1_fn, read_deck1, dir1 = load_agent("dev/submit003")
a2_fn, read_deck2, dir2 = load_agent("dev/submit002")

os.chdir(dir1)
deck1 = read_deck1()
os.chdir(dir2)
deck2 = read_deck2()
os.chdir(sample_dir)

print("Starting match: submit003 (Player 0 / First) vs submit002 (Player 1 / Second)")
obs_dict, _ = battle_start(deck1, deck2)

for turn in range(1, 15):
    if not obs_dict or (isinstance(obs_dict, dict) and obs_dict.get("is_finish")):
        print(f"Match Finished on turn {turn}! Winner: {obs_dict.get('winner') if isinstance(obs_dict, dict) else 'Unknown'}")
        break

    if isinstance(obs_dict, dict):
        curr_player = obs_dict.get("player", 0)
        options = obs_dict.get("select", {}).get("option", []) if isinstance(obs_dict.get("select"), dict) else []
    else:
        curr_player = getattr(obs_dict, "player", 0)
        select_obj = getattr(obs_dict, "select", None)
        options = getattr(select_obj, "option", []) if select_obj else []

    print(f"--- Step {turn} | Active Player: {curr_player} | Num Options: {len(options)} ---")
    if len(options) > 0:
        print(f"Sample Option 0: {options[0]}")

    if curr_player == 0 or curr_player is None:
        os.chdir(dir1)
        action = a1_fn(obs_dict)
        os.chdir(sample_dir)
        print(f"Action chosen by Player 0 (submit003): {action}")
    else:
        os.chdir(dir2)
        action = a2_fn(obs_dict)
        os.chdir(sample_dir)
        print(f"Action chosen by Player 1 (submit002): {action}")

    try:
        obs_dict = battle_select(action)
    except Exception as e:
        print(f"Exception on step {turn}: {e}")
        break

battle_finish()
