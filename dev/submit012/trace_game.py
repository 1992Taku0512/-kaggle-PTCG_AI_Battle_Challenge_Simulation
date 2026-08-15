import sys, os
project_root = os.getcwd()
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dev', 'submit012'))
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))

from cg.game import battle_start, battle_select
from cg.api import to_observation_class

d0 = [721]*4 + [722]*4 + [1158]*3 + [723]*4 + [1182]*3 + [1123]*4 + [1121]*4 + [1227]*9 + [3]*25
obs_dict, state = battle_start(d0, d0)
print('battle_start obs_dict:', obs_dict)

if obs_dict:
    obs = to_observation_class(obs_dict)
    print('obs.select:', obs.select)
    if obs.select:
        print('options:', len(obs.select.option))
        for i, opt in enumerate(obs.select.option):
            print(f'  opt {i}: type={getattr(opt, "type", None)}, text={opt}')
