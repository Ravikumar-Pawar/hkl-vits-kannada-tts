"""
HKL-VITS Hybrid Kannada TTS

Model Generation Pipeline

Flow:

Dataset Download
        |
Dataset Preparation
        |
HKL Training
        |
Checkpoint Generation

Run:

python generate_model.py
"""

import os

from app.pipelines.dataset_preparation import (
    KannadaTTSDatasetPipeline,
    )
from app.pipelines.training import (
    train_hkl_vits,
    )
# ======================================
# Paths
# ======================================

DATASET_DIR = os.path.join("dataset", "kannada_tts")

WAV_DIR = os.path.join(DATASET_DIR, "wavs")

TEXT_DIR = os.path.join(DATASET_DIR, "text")


# ======================================
# Generate Model
# ======================================


def generate_model():

    print("=" * 70)
    print("HKL-VITS KANNADA MODEL GENERATION")
    print("=" * 70)

    # ------------------------------
    # Step 1
    # Dataset
    # ------------------------------

    if not os.path.exists(TEXT_DIR):

        print("\nPreparing dataset...\n")

        dataset = KannadaTTSDatasetPipeline()

        dataset.run()

    else:

        print("\nDataset already prepared")

    # ------------------------------
    # Step 2
    # Training
    # ------------------------------

    print("\nStarting HKL training...\n")

    train_hkl_vits(wav_dir=WAV_DIR, text_dir=TEXT_DIR)

    print("\n" + "=" * 70)
    print("MODEL GENERATION COMPLETE")
    print("=" * 70)


# ======================================
# Main
# ======================================


if __name__ == "__main__":

    generate_model()
