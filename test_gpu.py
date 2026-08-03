import time
import torch
import torch.nn as nn
import torch.optim as optim
from notify_line import send_line_notification

def main():
    print("=" * 50)
    print("🚀 PyTorch GPU Training Test")
    print("=" * 50)

    # 1. CUDA / GPU Check
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {cuda_available}")

    if not cuda_available:
        msg = "❌ PyTorch GPUテスト失敗: CUDAが有効になっていません。"
        print(msg)
        send_line_notification(msg)
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU Device Name: {gpu_name}")
    print(f"VRAM Capacity:   {vram_gb:.2f} GB")
    print("-" * 50)

    # 2. Simple Neural Network Model on GPU
    class DummyPolicyValueNet(nn.Module):
        def __init__(self, input_dim=256, hidden_dim=512, action_dim=64):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_dim, action_dim)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            feat = self.backbone(x)
            policy_logits = self.policy_head(feat)
            state_value = self.value_head(feat)
            return policy_logits, state_value

    # Instantiate model and send to GPU
    model = DummyPolicyValueNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # 3. Create Synthetic Training Data on GPU (Batch size = 256)
    batch_size = 256
    epochs = 200
    inputs = torch.randn(batch_size, 256, device=device)
    target_actions = torch.randint(0, 64, (batch_size,), device=device)
    target_values = torch.randn(batch_size, 1, device=device)

    print(f"Starting GPU Training Benchmark ({epochs} Epochs, Batch Size={batch_size})...")
    start_time = time.time()

    # 4. Training Loop on GPU
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, values = model(inputs)
        
        loss_policy = policy_criterion(logits, target_actions)
        loss_value = value_criterion(values, target_values)
        total_loss = loss_policy + loss_value

        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Total Loss: {total_loss.item():.4f} | Device: {logits.device}")

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ GPU Training Completed Successfully in {elapsed_time:.3f} seconds!")
    print("=" * 50)

    # 5. Send LINE Notification
    line_msg = (
        f"🤖 【PTCG AI Battle】GPU動作検証完了通知！\n\n"
        f"・GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)\n"
        f"・CUDA: 正常認識 (torch {torch.__version__})\n"
        f"・ダミーモデル学習 (200 Epochs): {elapsed_time:.2f}秒で正常完了！\n\n"
        f"深層強化学習（PyTorch + CUDA）の環境構築が完了しました。"
    )
    send_line_notification(line_msg)

if __name__ == "__main__":
    main()
