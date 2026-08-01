import sys
import os

sample_dir = os.path.abspath("data/sample_submission/sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.game import battle_start, battle_select, battle_finish
from dev.submit003.main import read_deck_csv

deck = read_deck_csv()
obs_dict, _ = battle_start(deck, deck)

print("obs_dict type:", type(obs_dict))
print("obs_dict dict attributes:", [a for a in dir(obs_dict) if not a.startswith('_')])
if hasattr(obs_dict, '__dict__'):
    print("obs_dict __dict__:", obs_dict.__dict__)
else:
    print("obs_dict repr:", obs_dict)

battle_finish()
