"""
HKL-VITS Hybrid Kannada TTS

Kannada Phoneme Tokenizer
"""


class KannadaPhonemeTokenizer:

    def __init__(self):

        self.phonemes = [
            "<pad>",
            "<unk>",
            # vowels
            "a",
            "aa",
            "i",
            "ii",
            "u",
            "uu",
            "e",
            "ee",
            "o",
            "oo",
            # consonants
            "ka",
            "kha",
            "ga",
            "gha",
            "cha",
            "ja",
            "ṭa",
            "ṭha",
            "ɖa",
            "ɖha",
            "ta",
            "tha",
            "da",
            "dha",
            "na",
            "ṇa",
            "pa",
            "pha",
            "ba",
            "bha",
            "ma",
            "ya",
            "ra",
            "la",
            "va",
            "sha",
            "ṣa",
            "sa",
            "ha",
            # conjuncts
            "pra",
            "kra",
            "ksha",
            "tra",
            "gra",
            "ː",
        ]

        self.phoneme_to_id = {
            phoneme: index for index, phoneme in enumerate(self.phonemes)
        }

        self.id_to_phoneme = {
            index: phoneme for phoneme, index in self.phoneme_to_id.items()
        }

    def encode(self, text: str):

        tokens = text.split()

        return [
            self.phoneme_to_id.get(token, self.phoneme_to_id["<unk>"])
            for token in tokens
        ]

    def decode(self, ids):

        return " ".join(self.id_to_phoneme.get(idx, "<unk>") for idx in ids)
