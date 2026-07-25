"""
CNN-Based Kannada TTS
Model Architecture

KannadaCNNEncoder:
    Embedding
    → Stacked 1-D Depthwise-Separable Conv blocks (with residuals)
    → Prosody head  (pause / duration / pitch / energy)
    → MMS adapter   (projects to MMS hidden dim)

cnn-model/model.py
"""

import torch
import torch.nn as nn


# ============================================
# Depthwise-Separable Conv Block
# ============================================

class DSConvBlock(nn.Module):
    """1-D Depthwise-Separable Conv + residual + LayerNorm."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.depthwise  = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pointwise  = nn.Conv1d(channels, channels, 1)
        self.norm       = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, T, C)
        residual = x
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)          # (B, T, C)
        x = self.activation(x)
        x = self.norm(x + residual)
        return x


# ============================================
# CNN Encoder
# ============================================

class KannadaCNNEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim=256, cnn_layers=6, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DSConvBlock(hidden_dim, kernel_size) for _ in range(cnn_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)           # (B, T, H)
        x = self.input_proj(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        for block in self.blocks:
            x = block(x)

        return self.norm(x)                     # (B, T, H)


# ============================================
# Full CNN TTS Model
# ============================================

class KannadaCNNTTS(nn.Module):
    """
    CNN encoder + prosody head + MMS adapter.
    MMS backbone is frozen; only CNN layers are trained.
    """

    def __init__(self, mms_model, vocab_size, hidden_dim=256, cnn_layers=6, kernel_size=5):
        super().__init__()

        self.mms = mms_model
        for p in self.mms.parameters():
            p.requires_grad = False

        # MMS text-encoder hidden size
        mms_hidden = self.mms.config.hidden_size  # typically 192

        self.encoder  = KannadaCNNEncoder(vocab_size, hidden_dim, cnn_layers, kernel_size)
        self.prosody  = nn.Linear(hidden_dim, 4)          # pause / duration / pitch / energy
        self.adapter  = nn.Linear(hidden_dim, mms_hidden)

    def forward(self, input_ids, attention_mask=None):
        features = self.encoder(input_ids, attention_mask)  # (B, T, H)
        prosody  = self.prosody(features)                   # (B, T, 4)
        adapted  = self.adapter(features)                   # (B, T, mms_hidden)
        return {"features": features, "prosody": prosody, "adapted": adapted}
