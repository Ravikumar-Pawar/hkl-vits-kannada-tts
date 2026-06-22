"""
HKL-VITS Hybrid Kannada TTS

Kannada Text Normalizer Service
"""

import re
from typing import List

from app.config.hkl_config import HKLConfig


class KannadaTextNormalizer:

    def __init__(self, config: HKLConfig):

        self.config = config

    # ======================================
    # Whitespace cleanup
    # ======================================

    def normalize_whitespace(self, text: str) -> str:

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ======================================
    # Symbol replacement
    # ======================================

    def replace_symbols(self, text: str) -> str:

        if not self.config.expand_symbols:

            return text

        for symbol, replacement in self.config.symbol_map.items():

            text = text.replace(symbol, f" {replacement} ")

        return self.normalize_whitespace(text)

    # ======================================
    # Number expansion
    # ======================================

    def expand_number_token(self, token: str) -> str:

        if not self.config.expand_numbers:

            return token

        if not re.fullmatch(r"\d+", token):

            return token

        result = []

        for digit in token:

            result.append(self.config.digit_to_kn[digit])

        return " ".join(result)

    def expand_numbers(self, text: str) -> str:

        tokens = text.split()

        output = []

        for token in tokens:

            output.append(self.expand_number_token(token))

        return " ".join(output)

    # ======================================
    # Punctuation handling
    # ======================================

    def process_punctuation(self, text: str) -> str:

        if self.config.preserve_punctuation:

            return text

        text = re.sub(r"[^\w\sಀ-೿]", "", text)

        return text

    # ======================================
    # Main normalization pipeline
    # ======================================

    def normalize(self, text: str) -> str:

        text = self.normalize_whitespace(text)

        text = self.replace_symbols(text)

        text = self.expand_numbers(text)

        text = self.process_punctuation(text)

        text = self.normalize_whitespace(text)

        return text

    # ======================================
    # Tokenizer
    # ======================================

    def tokenize(self, text: str) -> List[str]:

        return self.normalize(text).split()
