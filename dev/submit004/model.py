import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual Block with Layer Normalization and ReLU activation"""
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    """Dual-Head Policy-Value Neural Network for PTCG AlphaZero Agent

    Inputs:
        state_tensor (torch.Tensor): Shape (Batch, State_Dim) - Output from StateEncoder (e.g. 269 dim)
    
    Outputs:
        policy_logits (torch.Tensor): Shape (Batch, Action_Dim) - Action probabilities/prior logits
        state_value (torch.Tensor): Shape (Batch, 1) - Estimated board win probability in [-1.0, +1.0]
    """
    def __init__(self, state_dim: int = 269, hidden_dim: int = 512, action_dim: int = 64, num_res_blocks: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input Projection Layer
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Backbone Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_res_blocks)])

        # Policy Head (Outputs prior logits for options)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Value Head (Outputs board evaluation in [-1, 1])
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if single sample

        feat = self.input_layer(x)
        for block in self.res_blocks:
            feat = block(feat)

        policy_logits = self.policy_head(feat)
        state_value = self.value_head(feat)

        return policy_logits, state_value


if __name__ == "__main__":
    # Smoke test model initialization and forward pass on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet().to(device)
    
    dummy_input = torch.randn(4, 269, device=device)  # Batch of 4
    logits, val = model(dummy_input)

    print(f"✅ AlphaZeroNet initialized on device: {device}")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Policy Shape: {logits.shape}")
    print(f"Value Shape:  {val.shape}")
