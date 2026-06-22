import os

import torch
from transformers import VitsModel, AutoTokenizer

from app.config.hkl_config import HKLConfig
from app.pipelines.training import HKLTrainerModel


class HKLVITSInference:

    def __init__(self, model, tokenizer, device):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls):

        config = HKLConfig()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        BASE_MODEL = "facebook/mms-tts-kan"

        CHECKPOINT = "checkpoints/" "hkl_vits_epoch_10.pt"

        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        print("Loading MMS...")

        mms = VitsModel.from_pretrained(BASE_MODEL).to(device)

        model = HKLTrainerModel(mms).to(device)

        if os.path.exists(CHECKPOINT):

            ckpt = torch.load(CHECKPOINT, map_location=device)

            model.load_state_dict(ckpt["model"])

            print("Checkpoint loaded")

        else:

            raise FileNotFoundError("Checkpoint missing")

        model.eval()

        return cls(model, tokenizer, device)

    def synthesize(self, text):

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():

            output = self.model.mms(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )

            wav = output.waveform.squeeze().cpu().numpy()

        return wav
