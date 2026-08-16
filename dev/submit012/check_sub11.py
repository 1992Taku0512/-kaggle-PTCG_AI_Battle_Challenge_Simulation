import sys, os
project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))
from cg.api import all_card_data
cards = {c.cardId: c for c in all_card_data()}
with open('dev/submit011/deck.csv') as f:
    ids = [int(l.strip()) for l in f if l.strip() and not l.startswith('#')]

out_path = 'dev/submit012/sub11_cards.txt'
with open(out_path, 'w') as out:
    for cid in sorted(set(ids)):
        c = cards.get(cid)
        count = ids.count(cid)
        hp = getattr(c, 'hp', None)
        stage = getattr(c, 'stage', None)
        out.write(f"ID {cid:4d}: {c.name:30s} x{count:2d} (HP={hp}, Stage={stage})\n")
