import sys, os
project_root = os.getcwd()
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dev', 'submit012'))
cg_dir = os.path.join(project_root, 'data', 'sample_submission', 'sample_submission')
sys.path.insert(0, cg_dir)

from cg.game import battle_start, battle_select
from cg.api import to_observation_class
from dev.common.trainer import load_deck_csv, extract_active_player, check_match_finish

d0 = load_deck_csv('dev/submit012/deck.csv')
obs_dict, state = battle_start(d0, d0)
print('Initial obs_dict:', obs_dict)

if obs_dict is not None:
    obs = to_observation_class(obs_dict)
    print('Initial options:', len(obs.select.option) if obs.select else None)

    for step in range(5):
        act = [0]
        print(f'Step {step} sending action:', act)
        try:
            obs_dict = battle_select(act)
            print(f'Step {step} result obs_dict:', obs_dict)
            if obs_dict is None:
                print('obs_dict is None!')
                break
            done, w = check_match_finish(obs_dict)
            if done:
                print(f'Match finished! winner={w}')
                break
        except Exception as e:
            import traceback
            traceback.print_exc()
            break
