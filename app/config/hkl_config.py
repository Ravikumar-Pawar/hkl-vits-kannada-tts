"""
HKL-VITS Hybrid Kannada TTS
Central Configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HKLConfig:

    mms_model_name: str = "facebook/mms-tts-kan"

    sample_rate: int = 16000

    # Text normalization

    expand_numbers: bool = True

    expand_symbols: bool = True

    preserve_punctuation: bool = True

    # Symbols

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

    # Number behaviour

    digit_by_digit_words: bool = False

    # numbers which should be spoken digit wise

    digit_context_words: List[str] = field(
        default_factory=lambda: ["otp", "pin", "code", "number", "mobile", "phone"]
    )

    # punctuation pause

    punctuation_map: Dict[str, str] = field(
        default_factory=lambda: {".": " ... ", ",": " , ", "?": " ... ", "!": " ... "}
    )
