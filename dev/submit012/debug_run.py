import sys, os
project_root = os.getcwd()
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))

from cg.game import battle_start, battle_select

log_path = os.path.join(os.path.dirname(__file__), "debug_output.txt")
with open(log_path, "w") as f:
    f.write("Starting test script...\n")
    d0 = [721]*4 + [722]*4 + [1158]*3 + [723]*4 + [1182]*3 + [1123]*4 + [1121]*4 + [1227]*9 + [3]*25
    obs_dict, state = battle_start(d0, d0)
    f.write(f"battle_start obs_dict keys: {list(obs_dict.keys()) if isinstance(obs_dict, dict) else obs_dict}\n")

    step = 0
    while obs_dict and step < 200:
        step += 1
        if "winner" in obs_dict:
            f.write(f"Step {step}: Winner -> {obs_dict['winner']}\n")
            break
        if "result" in obs_dict and obs_dict["result"] is not None:
            f.write(f"Step {step}: Result -> {obs_dict['result']}\n")
            break
        
        act = [0]
        try:
            obs_dict = battle_select(act)
        except Exception as e:
            f.write(f"Step {step}: Exception {e}\n")
            break
    f.write(f"Completed {step} steps.\n")
