import os
import random

from cg.api import Observation, SelectType, OptionType, to_observation_class

def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().strip().split("\n")
    deck = [int(line.strip()) for line in csv if line.strip()]
    return deck

def agent(obs_dict: dict) -> list[int]:
    """Rule-based Pokémon Trading Card Game Agent.
    
    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    
    options = obs.select.option
    num_options = len(options)
    if num_options == 0:
        return []
    
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    
    # メイン選択（自分のターンでの行動選択）
    if obs.select.type == SelectType.MAIN:
        # 優先度の設定
        # OptionType: ATTACK(13) > PLAY(7) > ATTACH(8) > EVOLVE(9) > ABILITY(10) > END(14)
        priority_order = [
            OptionType.ATTACK,
            OptionType.PLAY,
            OptionType.ATTACH,
            OptionType.EVOLVE,
            OptionType.ABILITY,
            OptionType.RETREAT,
            OptionType.END
        ]
        
        for preferred_type in priority_order:
            for idx, opt in enumerate(options):
                if opt.type == preferred_type:
                    return [idx]
        
        # 優先度に含まれない行動があれば0番目を選択
        return [0]
    
    # メイン以外の選択（カード選択や枚数選択など）
    # 要求される選択数範囲 [min_count, max_count] に収まる件数を選出
    if min_count == 0:
        return []
    
    # 必須選択の場合、min_count個の重複しないインデックスを選択
    k = max(min_count, 1)
    k = min(k, max_count, num_options)
    return list(range(k))

