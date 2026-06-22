"""
HKL-VITS Hybrid Kannada TTS

Class 9
Real Training Loop

T4 optimized
Mixed precision
Gradient accumulation
Checkpoint saving

app/pipelines/hkl_training_loop.py
"""

import os

import torch
import torch.nn as nn
from tqdm import tqdm

# ======================================
# Training Config
# ======================================


class TrainingConfig:

    CHECKPOINT_DIR = "checkpoints/hkl_vits"

    EPOCHS = 10

    GRAD_ACCUMULATION = 8

    LEARNING_RATE = 2e-4


# ======================================
# Loss Function
# ======================================


class HKLLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

    def phoneme_clarity_loss(self, features):

        if features.size(1) < 2:

            return torch.tensor(0.0, device=features.device)

        diff = features[:, 1:] - features[:, :-1]

        distance = torch.norm(diff, dim=-1)

        margin = 1.0

        loss = torch.relu(margin - distance).mean()

        return loss

    def forward(self, output):

        features = output["features"]

        prosody = output["prosody"]

        feature_loss = torch.mean(features**2)

        prosody_loss = torch.mean(prosody**2)

        clarity_loss = self.phoneme_clarity_loss(features)

        total = 0.4 * feature_loss + 0.2 * prosody_loss + 0.4 * clarity_loss

        return total


# ======================================
# Trainer
# ======================================


class HKLVITSTrainer:

    def __init__(self, model, loader, device="cuda"):

        self.model = model

        self.loader = loader

        self.device = device

        self.config = TrainingConfig()

        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

        self.criterion = HKLLoss()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.LEARNING_RATE,
        )

        if device == "cuda":

            self.scaler = torch.amp.GradScaler("cuda")

        else:

            self.scaler = None

    def save_checkpoint(self, epoch):

        path = os.path.join(self.config.CHECKPOINT_DIR, f"hkl_vits_epoch_{epoch}.pt")

        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

        print("Checkpoint saved:", path)

    def train(self):

        print("=" * 60)

        print("Starting HKL-VITS Training")

        print("=" * 60)

        global_step = 0

        for epoch in range(self.config.EPOCHS):

            self.model.train()

            epoch_loss = 0

            loop = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")

            self.optimizer.zero_grad()

            for step, batch in enumerate(loop):

                input_ids = batch["input_ids"]

                phoneme_ids = batch["phoneme_ids"]

                if self.device == "cuda":

                    with torch.amp.autocast("cuda"):

                        output = self.model(input_ids, phoneme_ids)

                        loss = self.criterion(output)

                        loss = loss / self.config.GRAD_ACCUMULATION

                    self.scaler.scale(loss).backward()

                else:

                    output = self.model(input_ids, phoneme_ids)

                    loss = self.criterion(output)

                    loss.backward()

                if (step + 1) % self.config.GRAD_ACCUMULATION == 0:

                    if self.device == "cuda":

                        self.scaler.step(self.optimizer)

                        self.scaler.update()

                    else:

                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    global_step += 1

                epoch_loss += loss.item()

                loop.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(self.loader)

            print(f"\nEpoch {epoch+1} Loss:", avg_loss)

            self.save_checkpoint(epoch + 1)

        print("=" * 60)

        print("HKL-VITS TRAINING COMPLETE")

        print("=" * 60)
