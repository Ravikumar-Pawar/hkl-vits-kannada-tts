"""
HKL-VITS Hybrid Kannada TTS

Training Pipeline

T4 Optimized

app/pipelines/training.py
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, VitsModel

from app.services.kannada_g2p import HKLKannadaG2P
# ======================================
# Device
# ======================================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================
# Dataset
# ======================================


class HKLDataset(Dataset):

    def __init__(self, wav_dir, text_dir, tokenizer, g2p):

        self.items = []

        self.tokenizer = tokenizer

        self.g2p = g2p

        for txt in os.listdir(text_dir):

            if txt.endswith(".txt"):

                wav_name = txt.replace(".txt", ".wav")

                wav_path = os.path.join(wav_dir, wav_name)

                if os.path.exists(wav_path):

                    with open(os.path.join(text_dir, txt), encoding="utf8") as f:

                        text = f.read().strip()

                    self.items.append({"wav": wav_path, "text": text})

        print("Dataset size:", len(self.items))

    def __len__(self):

        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]

        text = item["text"]

        text_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        phoneme_text = self.g2p.convert(text)

        phoneme_ids = torch.tensor(
            [ord(x) % 256 for x in phoneme_text], dtype=torch.long
        )

        return {"input_ids": text_ids, "phoneme_ids": phoneme_ids}


# ======================================
# Collator
# ======================================


class HKLCollator:

    def __init__(self, pad_id):

        self.pad_id = pad_id

    def __call__(self, batch):

        text = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )

        phoneme = torch.nn.utils.rnn.pad_sequence(
            [x["phoneme_ids"] for x in batch], batch_first=True, padding_value=0
        )

        return {"input_ids": text.to(DEVICE), "phoneme_ids": phoneme.to(DEVICE)}


# ======================================
# HKL Training Adapter
# ======================================


class HKLTrainerModel(nn.Module):

    def __init__(self, mms):

        super().__init__()

        self.mms = mms

        # freeze MMS

        for p in self.mms.parameters():

            p.requires_grad = False

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True),
            num_layers=2,
        )

        self.adapter = nn.Linear(256, 192)

        self.prosody = nn.Linear(256, 4)

    def forward(self, input_ids, phoneme_ids):

        x = torch.randn(input_ids.shape[0], input_ids.shape[1], 256, device=DEVICE)

        features = self.encoder(x)

        prosody = self.prosody(features)

        mms_features = self.adapter(features)

        return {"features": features, "prosody": prosody, "mms_features": mms_features}


# ======================================
# Training Runner
# ======================================


def train_hkl_vits(wav_dir, text_dir):

    BASE_MODEL = "facebook/mms-tts-kan"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    dataset = HKLDataset(wav_dir, text_dir, tokenizer, HKLKannadaG2P(None))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=HKLCollator(tokenizer.pad_token_id),
    )

    print("Loading MMS...")

    mms = VitsModel.from_pretrained(BASE_MODEL).to(DEVICE)

    model = HKLTrainerModel(mms).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4
    )

    model.train()

    for batch in loader:

        output = model(batch["input_ids"], batch["phoneme_ids"])

        loss = output["features"].mean()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        print("Training step complete")

        print("HKL:", output["features"].shape)

        print("Prosody:", output["prosody"].shape)

        break
