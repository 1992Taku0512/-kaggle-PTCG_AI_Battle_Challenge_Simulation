import sys, os
project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, 'data', 'sample_submission', 'sample_submission'))

from cg.game import battle_start

# 11 Basic Pokemons: 4 Kyogre (721), 4 Snover (722), 3 Delibird (757)
# 4 Evolution: 4 Mega Abomasnow ex (723)
# 1 ACE SPEC: 1 Maximum Belt (1158)
# 19 Trainers/Supporters: 4 Ultra Ball (1121), 4 Switch (1123), 4 Mega Signal (1145), 3 Boss (1182), 4 Lillie (1227)
# 25 Water Energies (3)
deck012 = [721]*4 + [722]*4 + [757]*3 + [723]*4 + [1158]*1 + [1121]*4 + [1123]*4 + [1145]*4 + [1182]*3 + [1227]*4 + [3]*25

print(f"Deck count: {len(deck012)}")
obs_dict, _ = battle_start(deck012, deck012)
print("battle_start result:", "SUCCESS (VALID DECK!)" if obs_dict is not None else "FAILED")
