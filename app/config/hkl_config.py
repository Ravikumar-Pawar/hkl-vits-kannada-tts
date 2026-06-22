"""
HKL-VITS Hybrid Kannada TTS
Central Configuration
app/config/hkl_config.py
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HKLConfig:

    # MMS / VITS
    mms_model_name: str = "facebook/mms-tts-kan"

    sample_rate: int = 16000

    # Linguistic pipeline

    use_g2p: bool = True

    use_grapheme_encoder: bool = True

    use_phoneme_encoder: bool = True

    # Prosody

    use_prosody_model: bool = True

    prosody_features: List[str] = field(
        default_factory=lambda: ["pause", "duration", "pitch", "energy"]
    )

    # Emotion

    use_emotion_embedding: bool = False

    supported_emotions: List[str] = field(
        default_factory=lambda: ["neutral", "happy", "sad", "angry", "questioning"]
    )

    # Text normalization

    expand_numbers: bool = True

    expand_symbols: bool = True

    preserve_punctuation: bool = True

    symbol_map: Dict[str, str] = field(
        default_factory=lambda: {"+": "ಪ್ಲಸ್", "=": "ಸಮಾನ", "%": "ಶೇಕಡಾ", "&": "ಮತ್ತು"}
    )

    digit_to_kn: Dict[str, str] = field(
        default_factory=lambda: {
            "0": "ಸೊನ್ನೆ",
            "1": "ಒಂದು",
            "2": "ಎರಡು",
            "3": "ಮೂರು",
            "4": "ನಾಲ್ಕು",
            "5": "ಐದು",
            "6": "ಆರು",
            "7": "ಏಳು",
            "8": "ಎಂಟು",
            "9": "ಒಂಬತ್ತು",
        }
    )

    # Training

    learning_rate: float = 2e-4

    batch_size: int = 2

    gradient_accumulation_steps: int = 8
