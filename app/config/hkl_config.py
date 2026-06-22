"""
HKL-VITS Hybrid Kannada TTS

Central Configuration
"""

import os

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HKLConfig:

    # ==================================================
    # Project Paths
    # ==================================================

    project_root: str = field(default_factory=lambda: os.getcwd())

    checkpoint_path: str = "checkpoints/hkl_vits_epoch_10.pt"

    output_dir: str = "outputs"

    cache_dir: str = "cache"

    # ==================================================
    # Model Configuration
    # ==================================================

    mms_model_name: str = "facebook/mms-tts-kan"

    model_revision: str = "main"

    load_checkpoint: bool = True

    # ==================================================
    # Device Configuration
    # ==================================================

    use_cuda: bool = True

    cuda_device: int = 0

    # ==================================================
    # Audio Configuration
    # ==================================================

    sample_rate: int = 16000

    audio_format: str = "wav"

    normalize_audio: bool = True

    # ==================================================
    # Tokenizer Configuration
    # ==================================================

    tokenizer_return_tensors: str = "pt"

    max_text_length: int = 512

    # ==================================================
    # Text Normalization
    # ==================================================

    normalize_text: bool = True

    expand_numbers: bool = True

    expand_symbols: bool = True

    preserve_punctuation: bool = True

    lowercase_text: bool = False

    # ==================================================
    # Symbol Expansion
    # ==================================================

    symbol_map: Dict[str, str] = field(
        default_factory=lambda: {
            "₹": "ರೂಪಾಯಿ",
            "$": "ಡಾಲರ್",
            "%": "ಶೇಕಡಾ",
            "+": "ಪ್ಲಸ್",
            "=": "ಸಮಾನ",
            "&": "ಮತ್ತು",
            "@": "ಅಟ್",
            "#": "ನಂಬರ್",
        }
    )

    # ==================================================
    # Number Processing
    # ==================================================

    digit_by_digit_words: bool = False

    convert_years: bool = True

    convert_decimals: bool = True

    digit_context_words: List[str] = field(
        default_factory=lambda: [
            "otp",
            "pin",
            "code",
            "number",
            "mobile",
            "phone",
        ]
    )

    # ==================================================
    # Punctuation Handling
    # ==================================================

    punctuation_map: Dict[str, str] = field(
        default_factory=lambda: {
            ".": " ... ",
            ",": " , ",
            "?": " ... ",
            "!": " ... ",
            ":": " , ",
            ";": " , ",
        }
    )

    # ==================================================
    # Inference Configuration
    # ==================================================

    normalize_before_inference: bool = True

    inference_no_grad: bool = True

    batch_size: int = 1

    # ==================================================
    # Training Configuration
    # ==================================================

    learning_rate: float = 1e-5

    epochs: int = 10

    save_every_epoch: bool = True

    # ==================================================
    # Logging Configuration
    # ==================================================

    log_level: str = "INFO"

    verbose: bool = True

    # ==================================================
    # Runtime Helpers
    # ==================================================

    def get_device(self):

        import torch

        if self.use_cuda and torch.cuda.is_available():

            return f"cuda:{self.cuda_device}"

        return "cpu"

    def checkpoint_exists(self):

        return os.path.exists(self.checkpoint_path)

    def ensure_directories(self):

        os.makedirs(self.output_dir, exist_ok=True)

        os.makedirs(self.cache_dir, exist_ok=True)
