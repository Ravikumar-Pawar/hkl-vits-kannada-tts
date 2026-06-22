"""
HKL-VITS Hybrid Kannada TTS

Kannada Text Normalizer
"""

import re
from typing import List

from indic_numtowords import num2words

from app.config.hkl_config import HKLConfig
from app.config.logger import setup_logger
logger = setup_logger(name=__name__)


class KannadaTextNormalizer:

    def __init__(self, config: HKLConfig):

        self.config = config

        logger.info("KannadaTextNormalizer initialized")

    # -------------------------------
    # whitespace
    # -------------------------------

    def normalize_whitespace(self, text: str) -> str:

        return re.sub(r"\s+", " ", text).strip()

    # -------------------------------
    # symbols
    # -------------------------------

    def replace_symbols(self, text: str) -> str:

        if not self.config.expand_symbols:

            return text

        for symbol, value in self.config.symbol_map.items():

            text = text.replace(symbol, f" {value} ")

        return text

    # -------------------------------
    # Kannada number engine
    # -------------------------------

    def number_to_kannada(self, value: int) -> str:

        try:

            return num2words(value, lang="kn")

        except Exception as e:

            logger.warning("Number conversion failed %s: %s", value, e)

            return str(value)

    # -------------------------------
    # number detection
    # -------------------------------

    def expand_number_token(self, token: str) -> str:

        original = token

        clean = token.replace(",", "")

        clean = re.sub(r"[₹$€£]", "", clean)

        suffix = ""

        if clean and clean[-1] in ".,!?":

            suffix = clean[-1]

            clean = clean[:-1]

        if not clean.isdigit():

            return original

        number = int(clean)

        result = self.number_to_kannada(number)

        return result + suffix

    def expand_numbers_text(self, text: str) -> str:

        if not self.config.expand_numbers:

            return text

        tokens = []

        for token in text.split():

            tokens.append(self.expand_number_token(token))

        return " ".join(tokens)

    # -------------------------------
    # punctuation
    # -------------------------------

    def process_punctuation(self, text: str) -> str:

        if not self.config.preserve_punctuation:

            return text

        for p, value in self.config.punctuation_map.items():

            text = text.replace(p, value)

        return text

    # -------------------------------
    # main pipeline
    # -------------------------------

    def normalize(self, text: str) -> str:

        if not text:

            return ""

        logger.debug("Original text: %s", text)

        text = self.normalize_whitespace(text)

        text = self.replace_symbols(text)

        text = self.expand_numbers_text(text)

        text = self.process_punctuation(text)

        text = self.normalize_whitespace(text)

        logger.debug("Normalized text: %s", text)

        return text

    # -------------------------------
    # tokenizer helper
    # -------------------------------

    def tokenize(self, text: str) -> List[str]:

        return self.normalize(text).split()
