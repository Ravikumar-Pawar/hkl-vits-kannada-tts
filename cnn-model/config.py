"""
CNN-Based Kannada TTS
Configuration

cnn-model/config.py
"""

import os
import torch


class CNNConfig:

    # Dataset (same OpenSLR dataset as HKL-VITS)
    DATA_DIR  = os.path.join("dataset", "kannada_tts")
    WAV_DIR   = os.path.join(DATA_DIR, "wavs")
    TEXT_DIR  = os.path.join(DATA_DIR, "text")
    TSV_NAME  = "metadata.tsv"
    ZIP_NAME  = "kannada_wavs.zip"

    WAV_ZIP_URL = "https://openslr.trmal.net/resources/79/kn_in_male.zip"
    TSV_URL     = "https://openslr.trmal.net/resources/79/line_index_male.tsv"

    # MMS backbone
    MMS_MODEL = "facebook/mms-tts-kan"

    # CNN architecture
    HIDDEN_DIM   = 256
    CNN_CHANNELS = 256
    CNN_LAYERS   = 6
    KERNEL_SIZE  = 5

    # Training
    EPOCHS           = 10
    BATCH_SIZE       = 4
    LEARNING_RATE    = 2e-4
    GRAD_ACCUMULATION = 4

    # Checkpoints
    CHECKPOINT_DIR = os.path.join("cnn-model", "checkpoints")

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
