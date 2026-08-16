import torch, os
ckpt_path = 'dev/submit012/checkpoints/latest_model.pt'
ckpt_dir = 'dev/submit012/checkpoints'
out_path = 'dev/submit012/status.txt'

with open(out_path, 'w') as f:
    f.write(f"Checkpoints in folder: {sorted(os.listdir(ckpt_dir))}\n")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            f.write(f"Saved Episode: {ckpt.get('episode')}\n")
            f.write(f"Best Winrate: {ckpt.get('best_winrate')}\n")
