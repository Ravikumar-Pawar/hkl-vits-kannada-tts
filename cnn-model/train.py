"""
CNN-Based Kannada TTS
Training Loop

Mixed precision | Gradient accumulation | Checkpoint saving

cnn-model/train.py
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VitsModel
from tqdm import tqdm

# Allow imports from project root
# Add cnn-model dir to path for sibling imports
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config  import CNNConfig
from dataset import KannadaCNNDataset, DatasetDownloader, cnn_collate
from model   import KannadaCNNTTS


# ============================================
# Loss
# ============================================

class CNNTTSLoss(nn.Module):

    def phoneme_clarity(self, features):
        if features.size(1) < 2:
            return torch.tensor(0.0, device=features.device)
        diff     = features[:, 1:] - features[:, :-1]
        distance = torch.norm(diff, dim=-1)
        return torch.relu(1.0 - distance).mean()

    def forward(self, output):
        feat    = output["features"]
        prosody = output["prosody"]
        return (
            0.4 * feat.pow(2).mean()
            + 0.2 * prosody.pow(2).mean()
            + 0.4 * self.phoneme_clarity(feat)
        )


# ============================================
# Trainer
# ============================================

class CNNTTSTrainer:

    def __init__(self, config: CNNConfig):
        self.config = config
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        print("Loading tokenizer and MMS backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MMS_MODEL)
        mms            = VitsModel.from_pretrained(config.MMS_MODEL).to(config.DEVICE)

        dataset = KannadaCNNDataset(config.TEXT_DIR, config.WAV_DIR, self.tokenizer)
        self.loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=cnn_collate,
            num_workers=0,
        )

        vocab_size  = self.tokenizer.vocab_size
        self.model  = KannadaCNNTTS(
            mms,
            vocab_size,
            config.HIDDEN_DIM,
            config.CNN_LAYERS,
            config.KERNEL_SIZE,
        ).to(config.DEVICE)

        self.criterion = CNNTTSLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.LEARNING_RATE,
        )
        self.scaler = torch.amp.GradScaler("cuda") if config.DEVICE == "cuda" else None

    def _save(self, epoch):
        path = os.path.join(self.config.CHECKPOINT_DIR, f"cnn_tts_epoch_{epoch}.pt")
        torch.save(
            {"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            path,
        )
        print("Checkpoint saved:", path)

    def train(self):
        print("=" * 60)
        print("CNN KANNADA TTS TRAINING")
        print(f"Device: {self.config.DEVICE} | Epochs: {self.config.EPOCHS}")
        print("=" * 60)

        for epoch in range(1, self.config.EPOCHS + 1):
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            loop = tqdm(self.loader, desc=f"Epoch {epoch}/{self.config.EPOCHS}")

            for step, batch in enumerate(loop):
                ids  = batch["input_ids"].to(self.config.DEVICE)
                mask = batch["attention_mask"].to(self.config.DEVICE)

                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        out  = self.model(ids, mask)
                        loss = self.criterion(out) / self.config.GRAD_ACCUMULATION
                    self.scaler.scale(loss).backward()
                else:
                    out  = self.model(ids, mask)
                    loss = self.criterion(out) / self.config.GRAD_ACCUMULATION
                    loss.backward()

                if (step + 1) % self.config.GRAD_ACCUMULATION == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

            print(f"Epoch {epoch} avg loss: {epoch_loss / len(self.loader):.4f}")
            self._save(epoch)

        print("=" * 60)
        print("CNN TTS TRAINING COMPLETE")
        print("=" * 60)


# ============================================
# Entry point
# ============================================

def train_cnn_tts():
    config = CNNConfig()

    # Download dataset if needed (same pipeline as HKL-VITS)
    if not os.path.exists(config.TEXT_DIR):
        print("Preparing dataset...")
        DatasetDownloader(config).run()

    trainer = CNNTTSTrainer(config)
    trainer.train()


if __name__ == "__main__":
    train_cnn_tts()
