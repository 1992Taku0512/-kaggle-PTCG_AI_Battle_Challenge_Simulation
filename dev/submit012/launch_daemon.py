import os
import sys
import subprocess
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
venv_python = os.path.join(project_root, ".venv", "bin", "python")
train_script = os.path.join(project_root, "dev", "submit012", "train.py")
stdout_log = os.path.join(project_root, "dev", "submit012", "train_stdout.log")

def launch():
    # Check if train.py is already running
    ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    if "dev/submit012/train.py" in ps.stdout:
        print("dev/submit012/train.py is ALREADY running!")
        for line in ps.stdout.splitlines():
            if "dev/submit012/train.py" in line:
                print(f"  Existing process: {line}")
        return

    print(f"Launching completely OS-detached daemon process for {train_script}...")
    with open(stdout_log, "a") as log_file:
        proc = subprocess.Popen(
            [venv_python, train_script],
            cwd=project_root,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,  # Fully detach from parent terminal/session (POSIX setsid)
        )
    print(f"✅ Daemon process launched successfully! PID: {proc.pid}")
    print(f"📄 Log output redirected to: {stdout_log}")

if __name__ == "__main__":
    launch()
