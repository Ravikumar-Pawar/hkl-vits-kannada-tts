"""
HKL-VITS Hybrid Kannada TTS

Dual Linguistic Encoder

Grapheme Transformer
+
Phoneme BiLSTM
+
Fusion Attention

app/models/hkl_linguistic_encoder.py
"""

import torch.nn as nn

# ======================================
# Grapheme Encoder
# ======================================


class KannadaGraphemeEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim=256, layers=2, heads=4):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = self.transformer(x)

        return x


# ======================================
# Phoneme Encoder
# ======================================


class KannadaPhonemeEncoder(nn.Module):

    def __init__(self, phoneme_vocab_size, hidden_dim=256):

        super().__init__()

        self.embedding = nn.Embedding(phoneme_vocab_size, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )

        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, phoneme_ids):

        x = self.embedding(phoneme_ids)

        x, _ = self.lstm(x)

        x = self.projection(x)

        return x


# ======================================
# Fusion Attention
# ======================================


class HKLFusionAttention(nn.Module):

    def __init__(self, hidden_dim=256, heads=4):

        super().__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, grapheme_features, phoneme_features):

        fused, _ = self.attention(
            query=grapheme_features, key=phoneme_features, value=phoneme_features
        )

        output = self.norm(grapheme_features + fused)

        return output


# ======================================
# Complete HKL Linguistic Encoder
# ======================================


class HKLVITSLinguisticEncoder(nn.Module):

    def __init__(self, grapheme_vocab, phoneme_vocab, hidden_dim=256):

        super().__init__()

        self.grapheme_encoder = KannadaGraphemeEncoder(
            vocab_size=grapheme_vocab, hidden_dim=hidden_dim
        )

        self.phoneme_encoder = KannadaPhonemeEncoder(
            phoneme_vocab_size=phoneme_vocab, hidden_dim=hidden_dim
        )

        self.fusion = HKLFusionAttention(hidden_dim=hidden_dim)

    def forward(self, grapheme_ids, phoneme_ids):

        grapheme_features = self.grapheme_encoder(grapheme_ids)

        phoneme_features = self.phoneme_encoder(phoneme_ids)

        fused = self.fusion(grapheme_features, phoneme_features)

        return fused
