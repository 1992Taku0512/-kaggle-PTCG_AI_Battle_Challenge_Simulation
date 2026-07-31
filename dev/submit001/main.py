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
    """Rule-based Pokémon Trading Card Game Agent with strict validation rules.
    
    Returns:
        list[int]: A list of option indices that comply with all environment constraints.
    """
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        
        options = obs.select.option
        num_options = len(options)
        if num_options == 0:
            return []
        
        min_count = obs.select.minCount
        max_count = obs.select.maxCount
        
        # 1. メイン選択（自分のターンでの主な選択行動）
        if obs.select.type == SelectType.MAIN:
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
            return [0]
        
        # 2. サブ選択（カード指定・エネルギー指定・対象選定など）
        # min_countが0（任意選択）の場合
        if min_count == 0:
            # 安全のためパス（空のリスト）を返す
            return []
        
        # min_count > 0 の必須選択の場合
        # minCount ～ maxCount の範囲内で、かつ重複しないインデックスを選択する
        k = max(min_count, 1)
        k = min(k, max_count, num_options)
        
        selected_indices = list(range(k))
        return selected_indices

    except Exception:
        # 万が一予期せぬ計算例外が発生した場合は、最も安全なフォールバック行動を返す
        try:
            select_info = obs_dict.get("select", {})
            min_c = select_info.get("minCount", 0)
            max_c = select_info.get("maxCount", 1)
            opts = select_info.get("option", [])
            num_opts = len(opts)
            
            if num_opts == 0 or min_c == 0:
                return []
            k = min(max(min_c, 1), max_c, num_opts)
            return list(range(k))
        except Exception:
            return [0]


