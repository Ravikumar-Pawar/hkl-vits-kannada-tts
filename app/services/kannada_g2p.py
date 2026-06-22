"""
HKL-VITS Hybrid Kannada TTS

Kannada Grapheme To Phoneme Engine
"""

from typing import List

from app.config.hkl_config import HKLConfig


class HKLKannadaG2P:

    def __init__(self, config: HKLConfig = None):

        self.config = config

        # ======================================
        # Independent vowels
        # ======================================

        self.vowels = {
            "ಅ": "a",
            "ಆ": "aa",
            "ಇ": "i",
            "ಈ": "ii",
            "ಉ": "u",
            "ಊ": "uu",
            "ಋ": "ru",
            "ಎ": "e",
            "ಏ": "ee",
            "ಐ": "ai",
            "ಒ": "o",
            "ಓ": "oo",
            "ಔ": "au",
        }

        # ======================================
        # Kannada consonants
        # ======================================

        self.consonants = {
            "ಕ": "ka",
            "ಖ": "kha",
            "ಗ": "ga",
            "ಘ": "gha",
            "ಚ": "cha",
            "ಛ": "chha",
            "ಜ": "ja",
            "ಝ": "jha",
            "ಟ": "ʈa",
            "ಠ": "ʈha",
            "ಡ": "ɖa",
            "ಢ": "ɖha",
            "ತ": "ta",
            "ಥ": "tha",
            "ದ": "da",
            "ಧ": "dha",
            "ಪ": "pa",
            "ಫ": "pha",
            "ಬ": "ba",
            "ಭ": "bha",
            "ಮ": "ma",
            "ಯ": "ya",
            "ರ": "ra",
            "ಲ": "la",
            "ವ": "va",
            "ಳ": "ɭa",
            "ಣ": "ɳa",
            "ನ": "na",
            "ಶ": "sha",
            "ಷ": "sha",
            "ಸ": "sa",
            "ಹ": "ha",
        }

        # ======================================
        # Vowel marks
        # ======================================

        self.vowel_marks = {
            "ಾ": "aa",
            "ಿ": "i",
            "ೀ": "ii",
            "ು": "u",
            "ೂ": "uu",
            "ೃ": "ru",
            "ೆ": "e",
            "ೇ": "ee",
            "ೈ": "ai",
            "ೊ": "o",
            "ೋ": "oo",
            "ೌ": "au",
        }

        # ======================================
        # Common conjuncts
        # ======================================

        self.conjuncts = {
            "ಕ್ಷ": "ksha",
            "ಜ್ಞ": "jna",
            "ಪ್ರ": "pra",
            "ಕ್ರ": "kra",
            "ಗ್ರ": "gra",
            "ತ್ರ": "tra",
            "ದ್ರ": "dra",
            "ಶ್ರ": "shra",
            "ಸ್ತ": "sta",
            "ಸ್ವ": "sva",
            "ಸ್ಕ": "ska",
        }

    # ======================================
    # Word conversion
    # ======================================

    def convert_word(self, word: str) -> str:

        phonemes = []

        i = 0

        if word in self.conjuncts:

            return self.conjuncts[word]

        while i < len(word):

            ch = word[i]

            if ch in self.vowels:

                phonemes.append(self.vowels[ch])

            elif ch in self.consonants:

                base = self.consonants[ch]

                # gemination

                if i + 2 < len(word) and word[i + 1] == "್" and word[i + 2] == ch:

                    phonemes.append(base.replace("a", "") + "ːa")

                    i += 2

                # consonant + vowel mark

                elif i + 1 < len(word) and word[i + 1] in self.vowel_marks:

                    phonemes.append(
                        base.replace("a", "") + self.vowel_marks[word[i + 1]]
                    )

                    i += 1

                else:

                    phonemes.append(base)

            elif ch == "ಂ":

                phonemes.append("m")

            elif ch == "ಃ":

                phonemes.append("h")

            i += 1

        return " ".join(phonemes)

    # ======================================
    # Sentence conversion
    # ======================================

    def convert_sentence(self, text: str) -> List[str]:

        words = text.split()

        return [self.convert_word(word) for word in words]
