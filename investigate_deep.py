import sys
import os
import traceback

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

print("=" * 80)
print("🔍 EXCEPTION TRACEBACK INVESTIGATION")
print("=" * 80)

obs_dict, _ = battle_start(deck1, deck2)
turn_count = 0

while obs_dict is not None and not obs_dict.get("is_finish") and turn_count < 20:
    turn_count += 1
    curr_player = obs_dict.get("player", 0) if isinstance(obs_dict, dict) else getattr(obs_dict, "player", 0)
    actor_name = "submit003" if curr_player == 0 else "submit002"
    actor_fn = a1_fn if curr_player == 0 else a2_fn
    actor_dir = dir1 if curr_player == 0 else dir2

    os.chdir(actor_dir)
    action = actor_fn(obs_dict)
    os.chdir(sample_dir)

    print(f"Step {turn_count:2d} | Player {curr_player} ({actor_name}) action: {action}")

    try:
        obs_dict = battle_select(action)
    except Exception as e:
        print(f"\n❌ EXCEPTION DETECTED on step {turn_count} by {actor_name}: {e}")
        print("Detailed Traceback:")
        traceback.print_exc()
        break

battle_finish()
