import sys, os
project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))

from cg.game import battle_start
from cg.api import all_card_data

all_cids = {c.cardId: c.name for c in all_card_data()}
res_file = os.path.join(os.path.dirname(__file__), "cids_res.txt")

with open(res_file, "w") as out:
    sub12_ids = [721, 722, 1158, 723, 1182, 1123, 1121, 1227, 3]
    for cid in sub12_ids:
        if cid in [1182, 1123, 1121, 1227, 3]:
            test_deck = [721]*20 + [cid]*15 + [3]*25
        else:
            test_deck = [cid]*30 + [3]*30
        if len(test_deck) == 60:
            try:
                o, _ = battle_start(test_deck, test_deck)
                res = "OK" if o is not None else "FAILED"
                out.write(f"Card ID {cid:4d} ({all_cids.get(cid, '?')}): {res}\n")
            except Exception as e:
                out.write(f"Card ID {cid:4d}: EXCEPTION {e}\n")
