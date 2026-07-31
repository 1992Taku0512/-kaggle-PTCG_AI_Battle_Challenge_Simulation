# eval_local.py 実装・検証レポート

ローカル環境でエージェント同士を多対戦させて統計（勝率・先後別勝率・平均ターン数・エラー発生率）を算出する評価用スクリプト [eval_local.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/eval_local.py) の作成が完了しました。

---

## 成果物

- **[eval_local.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/eval_local.py)**: ローカル評価用スクリプト

---

## 主な機能と使い方

1. **先攻・後攻の公平な対戦**:
   - `--num-games` で指定した試合数の半分ずつ、先攻・後攻を交互に入れ替えて評価します。
2. **勝率・統計情報の可視化**:
   - Agent1 vs Agent2 の全体の勝率
   - 先攻時勝率 / 後攻時勝率
   - 平均ターン数 / 引き分け・未終了数 / エラー理由内訳
3. **進行状況のプログレスバー (`tqdm`)** 表示。

### コマンド使用例

```bash
# dev/submit001 と サンプルエージェント を20試合対戦評価
uv run eval_local.py --agent1 dev/submit001 --agent2 data/sample_submission/sample_submission --num-games 20
```
