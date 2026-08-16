import sys, os
project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))

from cg.game import battle_start

def check_deck_file(path):
    with open(path) as f:
        deck = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]
    o, _ = battle_start(deck, deck)
    print(f"Deck {path}: len={len(deck)}, battle_start result: {'OK' if o is not None else 'FAILED'}")

check_deck_file('dev/submit011/deck.csv')
check_deck_file('dev/submit012/deck.csv')
