"""
HKL-VITS Hybrid Kannada TTS

Kannada Prosody Predictor

Predicts:
- pause
- duration
- pitch
- energy

app/models/kannada_prosody_model.py
"""

import torch.nn as nn


class KannadaProsodyModel(nn.Module):

    def __init__(self, hidden_dim=256, prosody_dim=4):

        super().__init__()

        # ======================================
        # BiLSTM Encoder
        # ======================================

        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # ======================================
        # Feature Projection
        # ======================================

        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # ======================================
        # Prosody Output
        #
        # 0 pause
        # 1 duration
        # 2 pitch
        # 3 energy
        #
        # ======================================

        self.output = nn.Linear(hidden_dim, prosody_dim)

    def forward(self, x):

        x, _ = self.encoder(x)

        x = self.projection(x)

        prosody = self.output(x)

        return {
            "pause": prosody[:, :, 0],
            "duration": prosody[:, :, 1],
            "pitch": prosody[:, :, 2],
            "energy": prosody[:, :, 3],
        }
