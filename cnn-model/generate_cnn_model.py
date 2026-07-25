"""
CNN-Based Kannada TTS
Entry Point

Flow:
    Dataset Download/Preparation
            |
    CNN Model Training
            |
    Checkpoint Generation

Run:
    python generate_cnn_model.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train import train_cnn_tts

if __name__ == "__main__":
    print("=" * 70)
    print("CNN KANNADA TTS — MODEL GENERATION")
    print("=" * 70)
    train_cnn_tts()
    print("\nDone. Checkpoints saved in cnn-model/checkpoints/")
