"""
CNN-Based Kannada TTS
Inference

Loads trained CNN checkpoint + MMS backbone and generates speech.

cnn-model/inference.py
"""

import os
import sys

import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, VitsModel

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import CNNConfig
from model  import KannadaCNNTTS


class CNNTTSInference:

    def __init__(self, checkpoint_path: str = None, config: CNNConfig = None):
        self.config = config or CNNConfig()
        self.device = self.config.DEVICE

        print("Loading tokenizer and MMS...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MMS_MODEL)
        self.mms       = VitsModel.from_pretrained(self.config.MMS_MODEL).to(self.device)

        vocab_size  = self.tokenizer.vocab_size
        self.model  = KannadaCNNTTS(
            self.mms,
            vocab_size,
            self.config.HIDDEN_DIM,
            self.config.CNN_LAYERS,
            self.config.KERNEL_SIZE,
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print("Checkpoint loaded:", checkpoint_path)
        else:
            print("No checkpoint — using untrained CNN encoder with pretrained MMS backbone")

        self.model.eval()

    @torch.no_grad()
    def synthesize(self, text: str, output_path: str = "output.wav") -> str:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).to(self.device)

        # CNN encoder features (used for prosody / future conditioning)
        cnn_out = self.model(enc.input_ids, enc.attention_mask)
        print(f"CNN features: {cnn_out['features'].shape} | Prosody: {cnn_out['prosody'].shape}")

        # MMS speech generation
        mms_out = self.mms(**enc)
        audio   = mms_out.waveform.squeeze().cpu().numpy()

        sf.write(output_path, audio, samplerate=16000)
        print("Audio saved:", output_path)
        return output_path


# ============================================
# Quick test
# ============================================

if __name__ == "__main__":
    config = CNNConfig()

    # Use latest checkpoint if available
    ckpt = None
    if os.path.exists(config.CHECKPOINT_DIR):
        pts = sorted([
            f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith(".pt")
        ])
        if pts:
            ckpt = os.path.join(config.CHECKPOINT_DIR, pts[-1])

    tts = CNNTTSInference(checkpoint_path=ckpt, config=config)
    tts.synthesize("ನಮಸ್ಕಾರ", output_path="cnn_output.wav")
