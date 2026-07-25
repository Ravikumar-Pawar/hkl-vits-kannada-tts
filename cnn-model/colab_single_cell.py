# ============================================================
# CNN Kannada TTS — Pure CNN, No MMS/VITS
# Single Colab Cell — Copy & Paste and Run
# Runtime → GPU (T4) recommended
# ============================================================

# ============================================================
# 1. Install
# ============================================================

import subprocess
subprocess.run(["pip", "install", "-q",
    "torch", "torchaudio", "librosa", "soundfile",
    "pandas", "requests", "tqdm", "numpy"
], check=True)

# ============================================================
# 2. Imports
# ============================================================

import os, zipfile, logging, json
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from IPython.display import Audio, display

# ============================================================
# 3. Logger
# ============================================================

def setup_logger(name="cnn_tts"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

log = setup_logger()

# ============================================================
# 4. GPU Check
# ============================================================

print("Torch     :", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device    :", DEVICE, f"({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")

# ============================================================
# 5. Config
# ============================================================

class Config:

    # Paths
    BASE        = "/content"
    DATA_DIR    = BASE + "/kannada_tts_dataset"
    WAV_DIR     = DATA_DIR + "/wavs"
    TEXT_DIR    = DATA_DIR + "/text"
    CKPT_DIR    = BASE + "/cnn_tts_checkpoints"
    VOCAB_PATH  = BASE + "/cnn_tts_vocab.json"

    # Dataset URLs (same OpenSLR dataset)
    WAV_ZIP_URL = "https://openslr.trmal.net/resources/79/kn_in_male.zip"
    TSV_URL     = "https://openslr.trmal.net/resources/79/line_index_male.tsv"
    ZIP_NAME    = "dataset.zip"
    TSV_NAME    = "metadata.tsv"

    # Audio
    SAMPLE_RATE = 22050
    N_FFT       = 1024
    HOP_LEN     = 256
    N_MELS      = 80
    MAX_WAV_LEN = 220500   # 10 seconds
    MAX_MEL_LEN = 860      # frames

    # Text
    MAX_TEXT_LEN = 150

    # CNN Model
    VOCAB_SIZE  = None     # set after vocab is built
    EMBED_DIM   = 256
    CNN_LAYERS  = 8
    KERNEL_SIZE = 5
    DROPOUT     = 0.1

    # Training
    EPOCHS            = 20
    BATCH_SIZE        = 8
    LEARNING_RATE     = 1e-3
    GRAD_ACCUMULATION = 4
    GRAD_CLIP         = 1.0

    DEVICE = DEVICE

cfg = Config()
os.makedirs(cfg.CKPT_DIR, exist_ok=True)

# ============================================================
# 6. Dataset Download & Preparation
# ============================================================

class DatasetPreparer:

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.DATA_DIR, exist_ok=True)
        os.makedirs(cfg.WAV_DIR,  exist_ok=True)

    def _download(self, url, path):
        log.info(f"Downloading {url}")
        r = requests.get(url, stream=True, timeout=60)
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f:
            for chunk in tqdm(r.iter_content(1024), total=total // 1024, unit="KB"):
                f.write(chunk)

    def run(self):
        zip_path = os.path.join(self.cfg.DATA_DIR, self.cfg.ZIP_NAME)
        tsv_path = os.path.join(self.cfg.DATA_DIR, self.cfg.TSV_NAME)

        if not os.path.exists(zip_path):
            self._download(self.cfg.WAV_ZIP_URL, zip_path)
        else:
            log.info("ZIP exists — skip")

        if not os.path.exists(tsv_path):
            self._download(self.cfg.TSV_URL, tsv_path)
        else:
            log.info("TSV exists — skip")

        wav_files = [f for f in os.listdir(self.cfg.WAV_DIR) if f.endswith(".wav")]
        if not wav_files:
            log.info("Extracting WAVs...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(self.cfg.WAV_DIR)
            log.info("Extraction done")
        else:
            log.info(f"WAVs ready: {len(wav_files)} files")

        if not os.path.exists(self.cfg.TEXT_DIR):
            os.makedirs(self.cfg.TEXT_DIR, exist_ok=True)
            df = pd.read_csv(tsv_path, sep="\t", header=None, names=["wav_file", "text"])
            for _, row in df.iterrows():
                p = os.path.join(self.cfg.TEXT_DIR, str(row["wav_file"]).strip() + ".txt")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(str(row["text"]).strip())
            log.info(f"Text files written: {len(df)}")
        else:
            log.info("Text files ready — skip")

# ============================================================
# 7. Character Vocabulary
# ============================================================

class KannadaVocab:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def build(self, text_dir, save_path):
        chars = set()
        for fname in os.listdir(text_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(text_dir, fname), encoding="utf-8") as f:
                chars.update(f.read().strip())

        specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        vocab    = specials + sorted(chars)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

        log.info(f"Vocab built: {len(self.char2idx)} chars → saved to {save_path}")
        return len(self.char2idx)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {int(i): c for c, i in self.char2idx.items()}
        log.info(f"Vocab loaded: {len(self.char2idx)} chars")
        return len(self.char2idx)

    def encode(self, text, max_len):
        ids = [self.char2idx.get(self.BOS, 1)]
        for c in text:
            ids.append(self.char2idx.get(c, self.char2idx[self.UNK]))
        ids.append(self.char2idx.get(self.EOS, 2))
        # pad / truncate
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids += [self.char2idx[self.PAD]] * (max_len - len(ids))
        return ids

# ============================================================
# 8. PyTorch Dataset
# ============================================================

class KannadaTTSDataset(Dataset):

    def __init__(self, cfg, vocab):
        self.cfg   = cfg
        self.vocab = vocab
        self.mel_transform = T.MelSpectrogram(
            sample_rate = cfg.SAMPLE_RATE,
            n_fft       = cfg.N_FFT,
            hop_length  = cfg.HOP_LEN,
            n_mels      = cfg.N_MELS,
        )
        self.items   = []
        skipped      = 0

        for fname in sorted(os.listdir(cfg.TEXT_DIR)):
            if not fname.endswith(".txt"):
                continue
            wav_path = os.path.join(cfg.WAV_DIR, fname.replace(".txt", ".wav"))
            if not os.path.exists(wav_path):
                continue
            with open(os.path.join(cfg.TEXT_DIR, fname), encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                skipped += 1
                continue
            self.items.append({"text": text, "wav": wav_path})

        log.info(f"Dataset: {len(self.items)} valid | {skipped} skipped")
        self._inspect(5)

    def _inspect(self, n):
        print("\n" + "=" * 65)
        print(f"  DATASET SAMPLE INSPECTION  (first {n} items)")
        print("=" * 65)
        for i, item in enumerate(self.items[:n]):
            ids = self.vocab.encode(item["text"], self.cfg.MAX_TEXT_LEN)
            print(f"  [{i}] text      : {item['text']}")
            print(f"       wav       : {os.path.basename(item['wav'])}")
            print(f"       chars     : {len(item['text'])}")
            print(f"       token_len : {len(ids)}")
            print(f"       input_ids : {ids[:12]} ...")
            print()
        print("=" * 65 + "\n")

    def _load_mel(self, wav_path):
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.cfg.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, self.cfg.SAMPLE_RATE)
        waveform = waveform.mean(0, keepdim=True)                   # mono
        waveform = waveform[:, :self.cfg.MAX_WAV_LEN]               # trim
        mel = self.mel_transform(waveform).squeeze(0)               # (n_mels, T)
        mel = torch.log(mel.clamp(min=1e-5))                        # log-mel
        # pad / trim to MAX_MEL_LEN
        T_mel = mel.shape[1]
        if T_mel >= self.cfg.MAX_MEL_LEN:
            mel = mel[:, :self.cfg.MAX_MEL_LEN]
        else:
            mel = F.pad(mel, (0, self.cfg.MAX_MEL_LEN - T_mel))
        return mel                                                   # (n_mels, MAX_MEL_LEN)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item    = self.items[idx]
        ids     = torch.tensor(
            self.vocab.encode(item["text"], self.cfg.MAX_TEXT_LEN),
            dtype=torch.long
        )
        mel     = self._load_mel(item["wav"])
        return {"input_ids": ids, "mel": mel}


def tts_collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "mel":       torch.stack([x["mel"]       for x in batch]),
    }

# ============================================================
# 9. Model — Pure CNN Text-to-Mel
# ============================================================

class DSConvBlock(nn.Module):
    """Depthwise-Separable 1-D Conv + residual + LayerNorm + GELU."""

    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.dw   = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pw   = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        r = x
        x = x.transpose(1, 2)
        x = self.dw(x)
        x = self.pw(x)
        x = x.transpose(1, 2)
        x = self.drop(self.act(x))
        return self.norm(x + r)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe  = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CNNTextEncoder(nn.Module):
    """Character embedding → stacked DS-Conv blocks → encoded sequence."""

    def __init__(self, vocab_size, embed_dim, cnn_layers, kernel_size, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.blocks  = nn.ModuleList([
            DSConvBlock(embed_dim, kernel_size, dropout) for _ in range(cnn_layers)
        ])
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        x = self.embed(input_ids)          # (B, T_text, E)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                # (B, T_text, E)


class CNNMelDecoder(nn.Module):
    """
    Upsamples encoded text sequence to mel-frame length,
    then refines with DS-Conv blocks → predicts mel spectrogram.
    """

    def __init__(self, embed_dim, n_mels, mel_len, text_len, cnn_layers, kernel_size, dropout):
        super().__init__()
        self.mel_len  = mel_len
        self.text_len = text_len
        # project embed_dim → n_mels for residual path
        self.upsample = nn.Linear(embed_dim, embed_dim)
        self.pos_enc  = PositionalEncoding(embed_dim)
        self.blocks   = nn.ModuleList([
            DSConvBlock(embed_dim, kernel_size, dropout) for _ in range(cnn_layers)
        ])
        self.norm     = nn.LayerNorm(embed_dim)
        self.proj     = nn.Linear(embed_dim, n_mels)

    def forward(self, encoded):
        # encoded: (B, T_text, E)
        # upsample T_text → mel_len via interpolation
        x = encoded.transpose(1, 2)                                 # (B, E, T_text)
        x = F.interpolate(x, size=self.mel_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2)                                       # (B, mel_len, E)
        x = self.pos_enc(x)
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x).transpose(1, 2)                         # (B, n_mels, mel_len)


class KannadaCNNTTS(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNTextEncoder(
            cfg.VOCAB_SIZE, cfg.EMBED_DIM,
            cfg.CNN_LAYERS, cfg.KERNEL_SIZE, cfg.DROPOUT
        )
        self.decoder = CNNMelDecoder(
            cfg.EMBED_DIM, cfg.N_MELS, cfg.MAX_MEL_LEN,
            cfg.MAX_TEXT_LEN, cfg.CNN_LAYERS, cfg.KERNEL_SIZE, cfg.DROPOUT
        )

    def forward(self, input_ids):
        enc = self.encoder(input_ids)       # (B, T_text, E)
        mel = self.decoder(enc)             # (B, n_mels, mel_len)
        return mel

# ============================================================
# 10. Loss
# ============================================================

class MelLoss(nn.Module):

    def forward(self, pred, target):
        # pred, target: (B, n_mels, T)
        mse  = F.mse_loss(pred, target)
        mae  = F.l1_loss(pred, target)
        return 0.5 * mse + 0.5 * mae

# ============================================================
# 11. Trainer
# ============================================================

class CNNTTSTrainer:

    def __init__(self, cfg, model, loader):
        self.cfg       = cfg
        self.model     = model.to(cfg.DEVICE)
        self.loader    = loader
        self.criterion = MelLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS
        )
        self.scaler    = torch.amp.GradScaler("cuda") if cfg.DEVICE == "cuda" else None

    def _save(self, epoch, loss):
        path = os.path.join(self.cfg.CKPT_DIR, f"cnn_tts_epoch_{epoch:02d}_loss{loss:.4f}.pt")
        torch.save({"epoch": epoch, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "loss": loss}, path)
        log.info(f"Checkpoint: {path}")
        return path

    def train(self):
        log.info("=" * 65)
        log.info("CNN KANNADA TTS — TRAINING (no MMS / no VITS)")
        log.info(f"Device: {self.cfg.DEVICE} | Epochs: {self.cfg.EPOCHS} | Batch: {self.cfg.BATCH_SIZE}")
        log.info(f"Vocab: {self.cfg.VOCAB_SIZE} | Mel: {self.cfg.N_MELS}×{self.cfg.MAX_MEL_LEN}")
        log.info("=" * 65)

        last_ckpt = None
        for epoch in range(1, self.cfg.EPOCHS + 1):
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            loop = tqdm(self.loader, desc=f"Epoch {epoch:02d}/{self.cfg.EPOCHS}")

            for step, batch in enumerate(loop):
                ids = batch["input_ids"].to(self.cfg.DEVICE)
                mel = batch["mel"].to(self.cfg.DEVICE)

                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        pred = self.model(ids)
                        loss = self.criterion(pred, mel) / self.cfg.GRAD_ACCUMULATION
                    self.scaler.scale(loss).backward()
                else:
                    pred = self.model(ids)
                    loss = self.criterion(pred, mel) / self.cfg.GRAD_ACCUMULATION
                    loss.backward()

                if (step + 1) % self.cfg.GRAD_ACCUMULATION == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()
            avg = epoch_loss / len(self.loader)
            log.info(f"Epoch {epoch:02d} | avg loss: {avg:.4f} | lr: {self.scheduler.get_last_lr()[0]:.6f}")
            last_ckpt = self._save(epoch, avg)

        log.info("TRAINING COMPLETE")
        return last_ckpt

# ============================================================
# 12. Inference — CNN mel → Griffin-Lim waveform
# ============================================================

class CNNTTSInference:

    def __init__(self, cfg, vocab, checkpoint_path=None):
        self.cfg    = cfg
        self.vocab  = vocab
        self.device = cfg.DEVICE

        self.model = KannadaCNNTTS(cfg).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            log.info(f"Loaded checkpoint: {checkpoint_path}")
        else:
            log.info("No checkpoint — running with random weights (for pipeline test)")
        self.model.eval()

        self.griffin_lim = T.GriffinLim(
            n_fft      = cfg.N_FFT,
            hop_length = cfg.HOP_LEN,
            n_iter     = 60,
        )
        self.inv_mel = T.InverseMelScale(
            n_stft      = cfg.N_FFT // 2 + 1,
            n_mels      = cfg.N_MELS,
            sample_rate = cfg.SAMPLE_RATE,
        )

    @torch.no_grad()
    def synthesize(self, text, output_path="/content/cnn_output.wav"):
        ids = torch.tensor(
            [self.vocab.encode(text, self.cfg.MAX_TEXT_LEN)],
            dtype=torch.long, device=self.device
        )
        log.info(f"Input text  : {text}")
        log.info(f"Input shape : {ids.shape}")

        mel_pred = self.model(ids)                          # (1, n_mels, mel_len)
        log.info(f"Mel shape   : {mel_pred.shape}")

        # exp to undo log, then invert mel → linear spectrogram → waveform
        mel_linear = mel_pred.squeeze(0).exp().cpu()        # (n_mels, mel_len)
        linear     = self.inv_mel(mel_linear)               # (n_fft//2+1, mel_len)
        waveform   = self.griffin_lim(linear)               # (mel_len * hop_len,)
        audio      = waveform.numpy()

        sf.write(output_path, audio, samplerate=self.cfg.SAMPLE_RATE)
        log.info(f"Audio saved : {output_path}  ({len(audio)/self.cfg.SAMPLE_RATE:.2f}s)")
        return output_path, audio

# ============================================================
# 13. Run Everything
# ============================================================

# ── Step 1: Dataset ──────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 1 — DATASET PREPARATION")
log.info("=" * 65)
DatasetPreparer(cfg).run()

# ── Step 2: Vocabulary ───────────────────────────────────────
log.info("=" * 65)
log.info("STEP 2 — VOCABULARY")
log.info("=" * 65)
vocab = KannadaVocab()
if os.path.exists(cfg.VOCAB_PATH):
    cfg.VOCAB_SIZE = vocab.load(cfg.VOCAB_PATH)
else:
    cfg.VOCAB_SIZE = vocab.build(cfg.TEXT_DIR, cfg.VOCAB_PATH)

# ── Step 3: Dataset & DataLoader ─────────────────────────────
log.info("=" * 65)
log.info("STEP 3 — DATASET & DATALOADER")
log.info("=" * 65)
dataset = KannadaTTSDataset(cfg, vocab)
loader  = DataLoader(
    dataset,
    batch_size  = cfg.BATCH_SIZE,
    shuffle     = True,
    collate_fn  = tts_collate,
    num_workers = 0,
)

# ── Step 4: Model ────────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 4 — MODEL")
log.info("=" * 65)
model     = KannadaCNNTTS(cfg)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Total params    : {total:,}")
log.info(f"Trainable params: {trainable:,}")

# ── Step 5: Train ────────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 5 — TRAINING")
log.info("=" * 65)
trainer   = CNNTTSTrainer(cfg, model, loader)
last_ckpt = trainer.train()

# ── Step 6: Inference ────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 6 — INFERENCE")
log.info("=" * 65)
tts = CNNTTSInference(cfg, vocab, checkpoint_path=last_ckpt)

test_sentences = [
    "ನಮಸ್ಕಾರ",
    "ಕನ್ನಡ ನಮ್ಮ ಭಾಷೆ",
    "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
]

for i, sentence in enumerate(test_sentences):
    out_path = f"/content/cnn_output_{i+1}.wav"
    path, audio = tts.synthesize(sentence, output_path=out_path)
    print(f"\nText: {sentence}")
    display(Audio(audio, rate=cfg.SAMPLE_RATE))

log.info("=" * 65)
log.info("ALL DONE")
log.info("=" * 65)
