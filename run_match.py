import os
import sys

# sample_submission ディレクトリに移動してエージェント間の対戦シミュレーションを実行するスクリプト
sample_dir = os.path.abspath("data/sample_submission/sample_submission")
sys.path.insert(0, sample_dir)

os.chdir(sample_dir)

from main import agent, read_deck_csv
from cg.game import battle_start, battle_select, battle_finish

def run_simulation():
    print("=== ポケモンカード AI 対戦シミュレーション開始 ===")
    
    # 1. 両プレイヤーのデッキ準備 (sample_submission/deck.csv を読み込み)
    deck0 = read_deck_csv()
    deck1 = read_deck_csv()
    print(f"プレイヤー1 デッキ枚数: {len(deck0)}枚")
    print(f"プレイヤー2 デッキ枚数: {len(deck1)}枚")
    
    # 2. バトル開始
    obs, start_data = battle_start(deck0, deck1)
    print(f"対戦開始！ (先攻/後攻 割り当て完了, BattlePtr: {start_data.battlePtr})")
    
    turn_count = 0
    max_turns = 1000
    
    # 3. ゲームループログ
    while obs is not None and turn_count < max_turns:
        turn_count += 1
        
        # 勝者判定・対戦終了チェック
        if isinstance(obs, dict) and obs.get("is_finish"):
            print(f"\n🎉 対戦終了！ ターン数: {turn_count}, 勝者: {obs.get('winner')}")
            break
            
        current_player = obs.get("player")
        
        # 各エージェントに状態 (obs) を渡して行動を取得
        action = agent(obs)
        
        if turn_count % 10 == 0 or turn_count < 5:
            print(f"ターン {turn_count:3d} | 手番: プレイヤー {current_player} | 選択可能オプション数: {len(obs.get('select', {}).get('option', [])) if obs.get('select') else 'N/A'} | エージェントの選択行動: {action}")
        
        # シミュレーターに行動を送信し、次の観測状態を取得
        try:
            obs = battle_select(action)
        except Exception as e:
            print(f"対戦終了 / 例外検知 (ターン {turn_count}): {type(e).__name__} - {e}")
            break
            
    # 4. バトル終了処理（メモリ解放）
    battle_finish()
    print("=== 対戦シミュレーション正常終了 ===")

if __name__ == "__main__":
    run_simulation()
