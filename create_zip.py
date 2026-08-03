import os
import zipfile

submit_dir = "dev/submit003"
cg_dir = "data/sample_submission/sample_submission/cg"
zip_path = os.path.join(submit_dir, "submission.zip")

# Agent files (flat, at root of zip)
agent_files = [
    "main.py",
    "deck.csv",
    "model.py",
    "state_encoder.py",
    "mcts.py",
    "model_weights.pt"
]

# cg module files (excluding __pycache__)
cg_files = [
    "__init__.py",
    "api.py",
    "cg.dll",
    "game.py",
    "libcg-arm64.so",
    "libcg.dylib",
    "libcg.so",
    "sim.py",
    "utils.py",
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in agent_files:
        fp = os.path.join(submit_dir, f)
        zf.write(fp, arcname=f)
        print(f"Added {f} ({os.path.getsize(fp)} bytes)")

    for f in cg_files:
        fp = os.path.join(cg_dir, f)
        zf.write(fp, arcname=os.path.join("cg", f))
        print(f"Added cg/{f} ({os.path.getsize(fp)} bytes)")

print(f"\nDONE: {zip_path} ({os.path.getsize(zip_path)} bytes)")
