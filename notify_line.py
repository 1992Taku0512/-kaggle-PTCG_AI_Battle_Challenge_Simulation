import os
import sys

def send_line_notification(message: str):
    """LINE notification is temporarily disabled by user request. Logs locally only."""
    print(f"📱 [LINE Disabled] Notification logged locally: {message}")

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "🤖 AntiGravity: Test notification"
    send_line_notification(msg)
