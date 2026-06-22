import os

import torch
from transformers import VitsModel, AutoTokenizer

from app.models.hkl_linguistic_encoder import HKLVITSLinguisticEncoder
from app.models.hkl_vits_wrapper import HKLVITS
from app.models.kannada_prosody_model import KannadaProsodyModel


class HKLModelLoader:

    def __init__(self, config):

        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_dir = "checkpoints"

        self.mms_dir = os.path.join(self.base_dir, "mms-tts-kan")

        self.checkpoint = os.path.join(self.base_dir, "hkl_vits_epoch_10.pt")

        self.model = None
        self.tokenizer = None

    def download_mms_if_missing(self):

        if os.path.exists(self.mms_dir):

            print("Using local MMS model")

            return

        print("Downloading MMS model...")

        os.makedirs(self.mms_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.config.mms_model_name)

        model = VitsModel.from_pretrained(self.config.mms_model_name)

        tokenizer.save_pretrained(self.mms_dir)

        model.save_pretrained(self.mms_dir)

        print("MMS saved locally")

    def load(self):

        print("Loading HKL-VITS...")

        self.download_mms_if_missing()

        self.tokenizer = AutoTokenizer.from_pretrained(self.mms_dir)

        linguistic_encoder = HKLVITSLinguisticEncoder(
            grapheme_vocab=200, phoneme_vocab=100, hidden_dim=256
        )

        prosody = KannadaProsodyModel(hidden_dim=256)

        self.model = HKLVITS(
            mms_model_name=self.mms_dir,
            linguistic_encoder=linguistic_encoder,
            prosody_model=prosody,
            hkl_hidden_dim=256,
        )

        checkpoint = torch.load(self.checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"], strict=False)

        self.model.to(self.device)

        self.model.eval()

        print("Checkpoint loaded")

        return (self.model, self.tokenizer, self.device)
