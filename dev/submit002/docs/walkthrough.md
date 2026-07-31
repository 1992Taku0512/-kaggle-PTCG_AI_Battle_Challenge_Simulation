# dev/submit002 実装完了ドキュメント

`cg` ライブラリの公式探索API（`search_begin`, `search_step`, `search_end`）を活用したルール検証付きエージェント **`dev/submit002`** の作成および動作確認が完了しました。

---

## 成果物

- **[dev/submit002/deck.csv](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit002/deck.csv)**: 対戦用デッキ（60枚）
- **[dev/submit002/main.py](file:///mnt/d/work/git/kaggle_PTCG_AI_Battle/dev/submit002/main.py)**: C++探索エンジンによる事前検証ロジックを搭載したエージェント

---

## 主な改良点

1. **事前ルール検証（Search API Validation）**:
   - `search_begin` で仮想盤面を構築し、各行動オプションを `search_step` に入力して判定。
   - ルール違反となる行動を事前に排除し、エラーとならない選択のみを選択。

2. **多段階フォールバック機能**:
   - メインフェーズ行動優先度: `ATTACK` > `PLAY` > `ATTACH` > `EVOLVE` > `ABILITY` > `RETREAT` > `END`
   - 万が一意図しない状態に陥った場合でも `try-except` で自動安全動作を行い、クラッシュを完全保護。

---

## LINE通知

- 実装・検証完了に合わせて LINE ポッシュ通知の送信を行いました (`send_line_notification`)。
