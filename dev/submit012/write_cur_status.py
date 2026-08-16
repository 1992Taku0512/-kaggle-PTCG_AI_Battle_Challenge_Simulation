import os, time, torch

ckpt_dir = 'dev/submit012/checkpoints'
out_path = 'dev/submit012/cur_status.txt'

with open(out_path, 'w') as f:
    files = sorted(os.listdir(ckpt_dir))
    f.write(f"Checkpoints: {files}\n")
    for fname in files:
        fp = os.path.join(ckpt_dir, fname)
        mtime = time.ctime(os.path.getmtime(fp))
        size_mb = os.path.getsize(fp) / (1024*1024)
        f.write(f"{fname:25s}: {size_mb:6.2f} MB | {mtime}\n")
    
    latest_path = os.path.join(ckpt_dir, 'latest_model.pt')
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            f.write(f"\nSaved Episode in latest_model.pt: {ckpt.get('episode')}\n")
            f.write(f"Best Winrate: {ckpt.get('best_winrate')}\n")
