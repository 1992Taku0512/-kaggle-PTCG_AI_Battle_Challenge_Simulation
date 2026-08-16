import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from state_encoder import encoder_size, decoder_size, num_words_encoder


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer calculating Cross-Attention between Option Queries and Board Context."""

    def __init__(self, d_model: int = 256, num_heads: int = 4, d_feedforward: int = 512):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_feedforward)
        self.fc2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # Cross Attention: x is Query (Options), encoder_out is Key and Value (Board Context)
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = F.relu(self.fc1(res))
        y = self.fc2(y)
        return self.norm2(res + y)


class Conv1DFeatureMixer(nn.Module):
    """Conv1D Pointwise Feature Mixer layer to extract local channel correlations."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Seq_Len, Batch, d_model) -> Transpose for Conv1D: (Batch, d_model, Seq_Len)
        seq_len, batch_size, d_model = x.shape
        x_t = x.permute(1, 2, 0)
        out_t = F.relu(self.conv1d(x_t))
        out = out_t.permute(2, 0, 1)  # Back to (Seq_Len, Batch, d_model)
        return self.norm(x + out)


class TransformerAlphaZeroNet(nn.Module):
    """Deep 8-Layer Conv1D + Transformer Hybrid Dual-Head Neural Network for PTCG AI.

    Architecture (8 Layers):
    1. EmbeddingBag Layers (encoder_bag: 22k -> 256, decoder_bag: 50k -> 256)
    2. Conv1D Feature Mixing Layer (Channel interaction & refinement)
    3. Transformer Encoder (2 Blocks, 4 Heads, 512 FFN) -> Board Context
    4. Decoder Cross-Attention (2 Blocks, 4 Heads, 512 FFN) -> Action Option Attn
    5. Dual-Head Linear Output (Value V(s) & Policy P(a|s))
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        d_feedforward: int = 512,
        num_layers_encoder: int = 2,
        num_layers_decoder: int = 2
    ):
        super().__init__()
        self.d_model = d_model

        # Layer 1: Embedding Bags
        self.encoder_bag = nn.EmbeddingBag(50000, d_model, mode="sum")
        self.decoder_bag = nn.EmbeddingBag(100000, d_model, mode="sum")

        # Layer 2: Conv1D Feature Mixers
        self.encoder_conv = Conv1DFeatureMixer(d_model)
        self.decoder_conv = Conv1DFeatureMixer(d_model)

        # Layer 3-4: Transformer Encoder (2 Blocks)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)

        # Layer 5-6: Transformer Decoder Cross-Attention (2 Blocks)
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder)
        ])

        # Layer 7-8: Dual-Head Outputs
        self.encoder_fc = nn.Linear(d_model, 1)
        self.decoder_fc = nn.Linear(d_model, 1)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Encoder Forward (Board State)
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.view(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)

        v = self.encoder_conv(v)
        encoder_out = self.encoder(v)

        val = self.encoder_fc(encoder_out)
        state_value = torch.tanh(val.mean(0))  # Value V(s) in [-1.0, +1.0]

        # 2. Decoder Forward (Action Options & Cross Attention)
        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.view(batch_size, -1, self.d_model).transpose(0, 1)

        p = self.decoder_conv(p)
        for dec_layer in self.decoder:
            p = dec_layer(p, encoder_out)

        p = self.decoder_fc(p)
        policy_logits = p.transpose(0, 1).contiguous().view(batch_size, -1)
        policy_scores = torch.tanh(policy_logits)  # Option Policy scores P(a|s)

        return state_value, policy_scores
