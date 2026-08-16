import os
import sys
import time
import zipfile
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
submit012_dir = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if submit012_dir not in sys.path:
    sys.path.insert(0, submit012_dir)

cg_path = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if cg_path not in sys.path:
    sys.path.insert(0, cg_path)

import torch

ckpt_100k = os.path.join(submit012_dir, "checkpoints", "model_ep100000.pt")
latest_ckpt = os.path.join(submit012_dir, "checkpoints", "latest_model.pt")
weights_path = os.path.join(submit012_dir, "model_weights.pt")
zip_path = os.path.join(project_root, "submit012.zip")
log_path = os.path.join(submit012_dir, "auto_submit.log")


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(log_path, "a") as f:
        f.write(formatted + "\n")


def check_training_finished():
    if os.path.exists(ckpt_100k):
        return True
    if os.path.exists(latest_ckpt):
        try:
            ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and ckpt.get("episode", 0) >= 100000:
                return True
        except Exception:
            pass
    return False


def extract_clean_weights(ckpt_source):
    log(f"Extracting clean model_weights.pt from {ckpt_source}...")
    ckpt = torch.load(ckpt_source, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    torch.save(state_dict, weights_path)
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    log(f"Clean model_weights.pt saved! Size: {size_mb:.2f} MB")


def build_zip():
    log("Building clean submit012.zip package...")
    files_to_zip = ["main.py", "model_weights.pt", "deck.csv", "mcts.py", "model.py", "state_encoder.py"]

    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            fp = os.path.join(submit012_dir, f)
            if os.path.exists(fp):
                zf.write(fp, arcname=f)
                log(f"  Added file: {f}")
            else:
                log(f"  ⚠️ Warning: missing file {f}")

        cg_dir = os.path.join(submit012_dir, "cg")
        for root, dirs, filenames in os.walk(cg_dir):
            for fn in filenames:
                if "__pycache__" in root or fn.endswith(".pyc"):
                    continue
                fp = os.path.join(root, fn)
                rel_path = os.path.relpath(fp, submit012_dir)
                zf.write(fp, arcname=rel_path)

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    log(f"✅ submit012.zip created successfully! Total size: {zip_mb:.2f} MB")


def test_agent_dry_run():
    log("Running local dry-run test on main.py...")
    cmd = [
        sys.executable, "-c",
        "import sys, os; sys.path.insert(0, 'dev/submit012'); from main import agent, read_deck_csv; "
        "deck = read_deck_csv(); assert len(deck)==60; "
        "dummy_obs = {}; ret = agent(dummy_obs); print('Dry run success! Returned:', len(ret))"
    ]
    res = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if res.returncode == 0:
        log("✅ Dry-run test PASSED cleanly!")
        return True
    else:
        log(f"❌ Dry-run test FAILED: {res.stderr}")
        return False


def submit_to_kaggle():
    log("Submitting submit012.zip to Kaggle competition 'pokemon-tcg-ai-battle'...")
    cmd = [
        "uv", "run", "kaggle", "competitions", "submit",
        "-c", "pokemon-tcg-ai-battle",
        "-f", zip_path,
        "-m", "submit012 100k Curriculum RL v7 (11 Basic Pokemons + Energy Readiness + LO Penalty)"
    ]
    res = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    log(f"Kaggle Submit Response (code {res.returncode}):\n{res.stdout}\n{res.stderr}")
    if res.returncode == 0:
        log("🎉 SUCCESSFULLY SUBMITTED submit012 TO KAGGLE!")
    else:
        log("❌ Kaggle submission failed. Please check logs.")


def main():
    log("Auto-submission monitor started. Executing packaging and submission...")
    target_ckpt = ckpt_100k if os.path.exists(ckpt_100k) else latest_ckpt
    extract_clean_weights(target_ckpt)
    build_zip()

    if test_agent_dry_run():
        submit_to_kaggle()


if __name__ == "__main__":
    main()
