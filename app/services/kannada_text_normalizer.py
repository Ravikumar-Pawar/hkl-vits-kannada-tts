"""
HKL-VITS Hybrid Kannada TTS

Kannada Text Normalizer
"""

import re
from typing import List

from indic_numtowords import num2words

from app.config.hkl_config import HKLConfig
class KannadaTextNormalizer:

    def __init__(self, config: HKLConfig):

        self.config = config

    # -------------------------------
    # whitespace
    # -------------------------------

    def normalize_whitespace(self, text):

        return re.sub(r"\s+", " ", text).strip()

    # -------------------------------
    # symbols
    # -------------------------------

    def replace_symbols(self, text):

        if not self.config.expand_symbols:

            return text

        for symbol, value in self.config.symbol_map.items():

            text = text.replace(symbol, f" {value} ")

        return text

    # -------------------------------
    # Kannada number engine
    # -------------------------------

    def number_to_kannada(self, value):

        try:

            return num2words(value, lang="kn")

        except Exception:

            return str(value)

    # -------------------------------
    # number detection
    # -------------------------------

    def expand_number_token(self, token):

        original = token

        currency = False

        if token.startswith("ರೂಪಾಯಿ"):

            currency = True

        # remove commas

        clean = token.replace(",", "")

        # remove currency symbols

        clean = re.sub(r"[₹$€£]", "", clean)

        # punctuation save

        suffix = ""

        if clean[-1:] in ".,!?":

            suffix = clean[-1]

            clean = clean[:-1]

        if not clean.isdigit():

            return original

        number = int(clean)

        result = self.number_to_kannada(number)

        return result + suffix

    def expand_numbers(self, text):

        output = []

        for token in text.split():

            output.append(self.expand_number_token(token))

        return " ".join(output)

    # -------------------------------
    # punctuation
    # -------------------------------

    def process_punctuation(self, text):

        for p, v in self.config.punctuation_map.items():

            text = text.replace(p, v)

        return text

    # -------------------------------
    # main
    # -------------------------------

    def normalize(self, text):

        text = self.normalize_whitespace(text)

        text = self.replace_symbols(text)

        text = self.expand_numbers(text)

        text = self.process_punctuation(text)

        text = self.normalize_whitespace(text)

        return text

    def tokenize(self, text) -> List[str]:

        return self.normalize(text).split()
