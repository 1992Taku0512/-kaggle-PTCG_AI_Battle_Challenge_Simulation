import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "0CADm0rED8dWUUKkjkBy/WzH6dmITpjyoqZsD3uGSO4sMl2gsP64paJACGiPyO+Zrs3Noy2RKrd7MHTq1FB7BgjQ45nFko606T4BLU9bmRUiKlUpHxs7MQD/QVf7Hp0nhYmevt5HXBRsDbvZ9DIoNQdB04t89/1O/w1cDnyilFU=")
LINE_USER_ID = os.getenv("LINE_USER_ID", "U45bdae4c5f77282b36dfc5e31658a3e7")

def send_line_notification(message: str):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("LINE notification sent successfully.")
    else:
        print(f"Failed to send LINE notification. Status: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "🤖 AntiGravity: Kaggle PTCG AI Battle タスク完了のテスト通知です！"
    send_line_notification(msg)
