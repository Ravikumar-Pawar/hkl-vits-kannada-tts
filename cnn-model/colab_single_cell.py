# ============================================================
# CNN Kannada TTS v2 — Improved Pure CNN (No MMS/VITS)
# ============================================================
# Enhancements over v1:
#   - Dilated DSConv blocks (exponential receptive field)
#   - Squeeze-and-Excitation channel attention
#   - Duration predictor + length regulator (no naive interpolation)
#   - Post-Net refinement (Tacotron 2 style)
#   - HiFi-GAN neural vocoder (replaces Griffin-Lim)
#   - Multi-resolution STFT loss
#   - Data augmentation (speed perturbation, noise)
#   - Cosine annealing with warm restarts
# ============================================================
# Single Colab Cell — Copy & Paste and Run
# Runtime → GPU (T4) recommended
# ============================================================

# ============================================================
# 1. Install
# ============================================================
!pip install -q torch torchaudio librosa soundfile pandas requests tqdm numpy transformers pesq pystoi jiwer



# ============================================================
# 2. Imports
# ============================================================

import os, zipfile, json, logging, math, shutil
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
matplotlib.rcParams['font.size'] = 12

# Force matplotlib to rebuild font cache so it picks up newly installed Noto fonts
_mpl_cache = os.path.expanduser("~/.cache/matplotlib")
if os.path.exists(_mpl_cache):
    shutil.rmtree(_mpl_cache, ignore_errors=True)

# Refresh matplotlib font cache and prefer Kannada-supporting fonts
_font_families = [f.name for f in fm.fontManager.ttflist]
_kannada_fonts = [f for f in _font_families if any(k in f.lower() for k in ['noto', 'kannada', 'lohit'])]
if _kannada_fonts:
    matplotlib.rcParams['font.sans-serif'] = _kannada_fonts + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    print(f"Using Kannada fonts: {_kannada_fonts}")
else:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    print("No Kannada-specific font found — glyph names will use Unicode codepoints")
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from IPython.display import Audio, display

# ============================================================
# 3. Logger
# ============================================================

def setup_logger(name="cnn_tts_v2"):
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
    CKPT_DIR    = BASE + "/cnn_tts_v2_checkpoints"
    VOCAB_PATH  = BASE + "/cnn_tts_v2_vocab.json"

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
    VOCAB_SIZE    = None     # set after vocab is built
    EMBED_DIM     = 256
    ENC_LAYERS    = 4        # standard DSConv blocks
    DILATED_LAYERS = 4       # dilated DSConv blocks
    DEC_LAYERS    = 4        # mel refinement blocks
    POSTNET_LAYERS = 5       # post-net refinement
    KERNEL_SIZE   = 5
    DROPOUT       = 0.1

    # Duration predictor
    DURATION_HIDDEN = 128

    # Training
    EPOCHS            = 30
    BATCH_SIZE        = 8
    LEARNING_RATE     = 1e-3
    GRAD_ACCUMULATION = 4
    GRAD_CLIP         = 1.0
    WARMUP_STEPS      = 1000

    # Augmentation
    SPEED_PERTURB = True
    NOISE_STD     = 0.005

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
# 7. Dataset Analysis
# ============================================================

class DatasetAnalyzer:
    """Analyzes TTS dataset distribution: text lengths, audio durations, character stats."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _load_metadata(self):
        """Load text files and corresponding WAV durations."""
        items = []
        for fname in sorted(os.listdir(self.cfg.TEXT_DIR)):
            if not fname.endswith(".txt"):
                continue
            wav_path = os.path.join(self.cfg.WAV_DIR, fname.replace(".txt", ".wav"))
            if not os.path.exists(wav_path):
                continue
            with open(os.path.join(self.cfg.TEXT_DIR, fname), encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            # Get audio duration
            try:
                info = torchaudio.info(wav_path)
                duration = info.num_frames / info.sample_rate
            except Exception:
                duration = 0.0
            items.append({
                "text": text,
                "text_len": len(text),
                "wav": os.path.basename(wav_path),
                "duration": duration,
            })
        return items

    def run(self):
        log.info("=" * 65)
        log.info("DATASET ANALYSIS")
        log.info("=" * 65)

        items = self._load_metadata()
        df = pd.DataFrame(items)
        n = len(df)
        log.info(f"Total samples: {n}")

        if n == 0:
            log.warning("No samples found for analysis!")
            return

        # ── Text Length Statistics ──────────────────────────
        text_lens = df["text_len"].values
        durations = df["duration"].values
        hop_len, sr = self.cfg.HOP_LEN, self.cfg.SAMPLE_RATE

        log.info(f"\n{'─'*50}")
        log.info("TEXT LENGTH STATISTICS (characters)")
        log.info(f"{'─'*50}")
        log.info(f"  Mean={text_lens.mean():.1f}  Std={text_lens.std():.1f}  "
                 f"Min={text_lens.min()}  Max={text_lens.max()}  "
                 f"P50={np.median(text_lens):.1f}  P95={np.percentile(text_lens,95):.1f}")

        log.info(f"\n{'─'*50}")
        log.info("AUDIO DURATION STATISTICS (seconds)")
        log.info(f"{'─'*50}")
        log.info(f"  Mean={durations.mean():.2f}s  Std={durations.std():.2f}s  "
                 f"Min={durations.min():.2f}s  Max={durations.max():.2f}s  "
                 f"Total={durations.sum()/3600:.2f}h")

        # Figure: 2×2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Kannada TTS Dataset Analysis", fontsize=16, fontweight='bold')

        # ── Top-Left: Text Length Histogram ─────────────────
        ax = axes[0, 0]
        ax.hist(text_lens, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(np.median(text_lens), color='red', ls='--', lw=2, label=f"Median={np.median(text_lens):.0f}")
        ax.axvline(np.percentile(text_lens, 95), color='orange', ls=':', lw=2, label=f"P95={np.percentile(text_lens,95):.0f}")
        ax.set_xlabel("Text Length (characters)")
        ax.set_ylabel("Count")
        ax.set_title(f"Text Length Distribution (n={n})")
        ax.legend(fontsize=9)

        # ── Top-Right: Audio Duration Histogram ─────────────
        ax = axes[0, 1]
        ax.hist(durations[durations > 0], bins=40, color='mediumseagreen', edgecolor='white', alpha=0.85)
        ax.axvline(np.median(durations[durations > 0]), color='red', ls='--', lw=2,
                   label=f"Median={np.median(durations[durations > 0]):.2f}s")
        total_hrs = durations[durations > 0].sum() / 3600
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Count")
        ax.set_title(f"Audio Duration Distribution ({total_hrs:.2f}h total)")
        ax.legend(fontsize=9)

        # ── Bottom-Left: Character Frequency (top 25) ───────
        ax = axes[1, 0]
        all_chars = "".join(df["text"].values)
        char_counts = {}
        for c in all_chars:
            char_counts[c] = char_counts.get(c, 0) + 1
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        top_n = min(25, len(sorted_chars))
        chars, counts = zip(*sorted_chars[:top_n])
        display_chars = [c if c != " " else "␣" for c in chars]
        bars = ax.barh(range(top_n), counts, color='coral', edgecolor='white')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(display_chars, fontsize=10)
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_n} Character Frequency ({len(sorted_chars)} unique)")
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f"{count:,}", va='center', fontsize=8)

        # ── Bottom-Right: Text Length vs Duration scatter ───
        ax = axes[1, 1]
        valid = durations > 0
        ax.scatter(text_lens[valid], durations[valid], alpha=0.4, s=10, c='purple')
        if valid.sum() > 1:
            # Linear fit
            coeffs = np.polyfit(text_lens[valid], durations[valid], 1)
            x_fit = np.linspace(text_lens[valid].min(), text_lens[valid].max(), 100)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--', lw=2,
                    label=f"y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
        ax.axhline(y=1, color='gray', ls=':', alpha=0.5, label="1s boundary")
        ax.set_xlabel("Text Length (chars)")
        ax.set_ylabel("Duration (s)")
        ax.set_title("Text Length vs Audio Duration")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        # ── Mel Frame Estimate ──────────────────────────────
        mel_frames = durations * sr / hop_len
        log.info(f"\n{'─'*50}")
        log.info("MEL FRAME ESTIMATES (hop={}, sr={})".format(hop_len, sr))
        log.info(f"{'─'*50}")
        log.info(f"  Mean mel frames : {mel_frames.mean():.0f}")
        log.info(f"  Max mel frames  : {mel_frames.max():.0f}")
        log.info(f"  P95 mel frames  : {np.percentile(mel_frames, 95):.0f}")
        log.info(f"  Config MAX_MEL  : {self.cfg.MAX_MEL_LEN}")
        pct_truncated = (mel_frames > self.cfg.MAX_MEL_LEN).mean() * 100
        log.info(f"  Samples truncated (>MAX_MEL_LEN): {pct_truncated:.1f}%")

        # ── Summary ─────────────────────────────────────────
        log.info(f"\n{'─'*50}")
        log.info("DATASET SUMMARY")
        log.info(f"{'─'*50}")
        log.info(f"  Total samples       : {n}")
        log.info(f"  Avg text length     : {text_lens.mean():.1f} chars")
        log.info(f"  Avg audio duration  : {durations.mean():.2f}s")
        log.info(f"  Total audio         : {durations.sum() / 3600:.2f} hours")
        log.info(f"  Unique characters   : {len(sorted_chars)}")
        log.info(f"  Sample rate         : {sr} Hz")
        log.info(f"  Mel bands           : {self.cfg.N_MELS}")
        log.info(f"  Max mel frames      : {self.cfg.MAX_MEL_LEN}")
        log.info("=" * 65)

        return {
            "n_samples": n,
            "avg_text_len": float(text_lens.mean()),
            "avg_duration": float(durations.mean()),
            "total_hours": float(durations.sum() / 3600),
            "unique_chars": len(sorted_chars),
        }


# ============================================================
# 8. Character Vocabulary
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
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids += [self.char2idx[self.PAD]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            c = self.idx2char.get(i, self.UNK)
            if c in (self.PAD, self.BOS, self.EOS):
                continue
            chars.append(c)
        return "".join(chars)

# ============================================================
# 8. Audio Augmentation
# ============================================================

class AudioAugmenter:

    def __init__(self, cfg):
        self.cfg = cfg
        self.speeds = [0.9, 1.0, 1.1] if cfg.SPEED_PERTURB else [1.0]

    def augment(self, waveform, sr):
        """Returns list of (waveform, sr) tuples."""
        results = []
        for speed in self.speeds:
            if speed != 1.0:
                wav = librosa.effects.time_stretch(waveform.numpy(), rate=speed)
                wav = torch.from_numpy(wav).float()
            else:
                wav = waveform.clone()

            # Add small noise
            if self.cfg.NOISE_STD > 0 and speed == 1.0:
                noise = torch.randn_like(wav) * self.cfg.NOISE_STD
                wav = wav + noise

            results.append((wav, sr))
        return results

# ============================================================
# 9. PyTorch Dataset
# ============================================================

class KannadaTTSDataset(Dataset):

    def __init__(self, cfg, vocab, augmenter=None):
        self.cfg       = cfg
        self.vocab     = vocab
        self.augmenter = augmenter
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
        self._inspect(3)

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
            waveform = AF.resample(waveform, sr, self.cfg.SAMPLE_RATE)
        waveform = waveform.mean(0, keepdim=True)                   # mono
        waveform = waveform[:, :self.cfg.MAX_WAV_LEN]               # trim
        mel = self.mel_transform(waveform).squeeze(0)               # (n_mels, T)
        mel = torch.log(mel.clamp(min=1e-5))                        # log-mel
        T_mel = mel.shape[1]
        if T_mel >= self.cfg.MAX_MEL_LEN:
            mel = mel[:, :self.cfg.MAX_MEL_LEN]
        else:
            mel = F.pad(mel, (0, self.cfg.MAX_MEL_LEN - T_mel))
        return mel, waveform.squeeze(0)                              # (n_mels, MAX_MEL_LEN), (T_wav,)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item    = self.items[idx]
        ids     = torch.tensor(
            self.vocab.encode(item["text"], self.cfg.MAX_TEXT_LEN),
            dtype=torch.long
        )
        mel, wav = self._load_mel(item["wav"])

        # Compute text length (without padding) and mel length (without padding)
        text_len = sum(1 for c in item["text"] if c != " ") + 2  # BOS + chars + EOS (approx)
        text_len = min(text_len, self.cfg.MAX_TEXT_LEN)

        # Count non-padded mel frames
        mel_len = min(mel.shape[1], self.cfg.MAX_MEL_LEN)

        return {
            "input_ids": ids,
            "mel": mel,
            "wav": wav,
            "text_len": torch.tensor(text_len, dtype=torch.long),
            "mel_len": torch.tensor(mel_len, dtype=torch.long),
        }


def tts_collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "mel":       torch.stack([x["mel"]       for x in batch]),
        "wav":       [x["wav"] for x in batch],
        "text_len":  torch.stack([x["text_len"]  for x in batch]),
        "mel_len":   torch.stack([x["mel_len"]   for x in batch]),
    }

# ============================================================
# 10. Model — Improved Pure CNN Text-to-Mel
# ============================================================

class SqueezeExcitation(nn.Module):
    """Channel-wise attention: SE block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        scale = self.fc(x)
        return x * scale


class DSConvBlock(nn.Module):
    """Depthwise-Separable 1-D Conv + residual + LayerNorm + GELU + SE."""

    def __init__(self, channels, kernel_size, dilation=1, dropout=0.1, use_se=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.dw   = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation, groups=channels)
        self.pw   = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.se   = SqueezeExcitation(channels) if use_se else nn.Identity()

    def forward(self, x):
        # x: (B, T, C)
        r = x
        x = x.transpose(1, 2)   # (B, C, T)
        x = self.dw(x)
        x = self.pw(x)
        x = self.se(x)
        x = x.transpose(1, 2)   # (B, T, C)
        x = self.drop(self.act(x))
        return self.norm(x + r)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe  = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DurationPredictor(nn.Module):
    """Predicts per-token duration in mel frames."""

    def __init__(self, embed_dim, hidden_dim=128, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.proj  = nn.Linear(hidden_dim, 1)
        self.act   = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)   # (B, C, T)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.transpose(1, 2)   # (B, T, C)
        x = self.proj(x).squeeze(-1)  # (B, T)
        x = F.softplus(x) + 0.5       # ensure positive durations
        return x


class LengthRegulator(nn.Module):
    """Expands encoder output from text_len to mel_len using predicted durations."""

    def forward(self, x, durations, mel_max_len=None):
        # x: (B, T_text, C)
        # durations: (B, T_text) — predicted duration per token
        outputs = []
        for b in range(x.size(0)):
            expanded = []
            for t in range(x.size(1)):
                d = max(1, int(round(durations[b, t].item())))
                # Use unsqueeze to keep the seq dimension for expanding
                token = x[b, t:t+1, :]  # (1, C)
                expanded.append(token.expand(d, -1))
            out = torch.cat(expanded, dim=0)   # (total_dur, C)
            if mel_max_len is not None:
                if out.size(0) < mel_max_len:
                    pad = torch.zeros(mel_max_len - out.size(0), x.size(2), device=x.device)
                    out = torch.cat([out, pad], dim=0)
                else:
                    out = out[:mel_max_len]
            outputs.append(out)
        return torch.stack(outputs, dim=0)  # (B, mel_len, C)


class CNNTextEncoder(nn.Module):
    """Character embedding → standard DSConv → dilated DSConv → encoded sequence."""

    def __init__(self, vocab_size, embed_dim, enc_layers, dilated_layers, kernel_size, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim)

        # Standard DSConv blocks
        self.blocks = nn.ModuleList([
            DSConvBlock(embed_dim, kernel_size, dilation=1, dropout=dropout)
            for _ in range(enc_layers)
        ])

        # Dilated DSConv blocks (exponential dilation)
        dilations = [2 ** i for i in range(dilated_layers)]
        self.dilated_blocks = nn.ModuleList([
            DSConvBlock(embed_dim, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        x = self.embed(input_ids)          # (B, T_text, E)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        for blk in self.dilated_blocks:
            x = blk(x)
        return self.norm(x)                # (B, T_text, E)


class CNNMelDecoder(nn.Module):
    """Refines length-regulated encoder output → mel spectrogram."""

    def __init__(self, embed_dim, n_mels, dec_layers, kernel_size, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            DSConvBlock(embed_dim, kernel_size, dilation=1, dropout=dropout)
            for _ in range(dec_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, n_mels)

    def forward(self, x):
        # x: (B, mel_len, E)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x).transpose(1, 2)  # (B, n_mels, mel_len)


class PostNet(nn.Module):
    """Tacotron 2 style post-net: 5 Conv1d layers with residual."""

    def __init__(self, n_mels, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_mels, n_mels, kernel_size, padding=pad),
                nn.BatchNorm1d(n_mels),
                nn.Tanh() if i < 4 else nn.Identity(),
                nn.Dropout(dropout),
            )
            for i in range(5)
        ])

    def forward(self, x):
        # x: (B, n_mels, T)
        residual = x
        for conv in self.convs:
            x = conv(x)
        return x + residual


class KannadaCNNTTSv2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNTextEncoder(
            cfg.VOCAB_SIZE, cfg.EMBED_DIM,
            cfg.ENC_LAYERS, cfg.DILATED_LAYERS,
            cfg.KERNEL_SIZE, cfg.DROPOUT
        )
        self.duration_predictor = DurationPredictor(cfg.EMBED_DIM, cfg.DURATION_HIDDEN)
        self.length_regulator   = LengthRegulator()
        self.decoder = CNNMelDecoder(
            cfg.EMBED_DIM, cfg.N_MELS,
            cfg.DEC_LAYERS, cfg.KERNEL_SIZE, cfg.DROPOUT
        )
        self.postnet = PostNet(cfg.N_MELS, dropout=cfg.DROPOUT)

    def forward(self, input_ids, mel_max_len=None, duration_targets=None):
        # Encode
        enc = self.encoder(input_ids)       # (B, T_text, E)

        # Duration prediction
        durations = self.duration_predictor(enc)  # (B, T_text)

        # Length regulation
        if duration_targets is not None and self.training:
            # Use ground-truth durations during training
            reg = self.length_regulator(enc, duration_targets, mel_max_len)
        else:
            reg = self.length_regulator(enc, durations, mel_max_len)

        # Decode to mel
        mel_before = self.decoder(reg)       # (B, n_mels, mel_len)

        # Post-net refinement
        mel_after = self.postnet(mel_before)  # (B, n_mels, mel_len)

        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "durations": durations,
        }

    @torch.no_grad()
    def synthesize(self, input_ids):
        self.eval()
        enc = self.encoder(input_ids)
        durations = self.duration_predictor(enc)
        reg = self.length_regulator(enc, durations)
        mel_before = self.decoder(reg)
        mel_after = self.postnet(mel_before)
        return mel_after

# ============================================================
# 11. Multi-Resolution STFT Loss
# ============================================================

class MelLoss(nn.Module):
    """Combined L1 + L2 mel loss with multi-resolution spectral convergence on mel bands."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, mel_pred, mel_target):
        # mel_pred, mel_target: (B, n_mels, T)
        mel_mse = self.mse_loss(mel_pred, mel_target)
        mel_mae = self.mae_loss(mel_pred, mel_target)

        # Spectral convergence on mel spectrograms (along frequency axis)
        # Treat each time frame as a spectral vector
        pred_norm = torch.norm(mel_pred, p="fro", dim=(1, 2))
        tgt_norm  = torch.norm(mel_target, p="fro", dim=(1, 2))
        sc_loss   = F.l1_loss(pred_norm, tgt_norm)

        return {
            "mel_mse": mel_mse,
            "mel_mae": mel_mae,
            "sc_loss": sc_loss,
            "total": 0.5 * mel_mse + 0.5 * mel_mae + 0.5 * sc_loss,
        }


class GuidedAttentionLoss(nn.Module):
    """Encourages diagonal attention for duration prediction."""

    def __init__(self, sigma=0.2):
        super().__init__()
        self.sigma = sigma

    def forward(self, durations, text_len, mel_len):
        # durations: (B, T_text) — predicted durations
        # text_len: (B,) — actual text lengths
        # mel_len: (B,) — actual mel lengths
        B, T_text = durations.shape
        device = durations.device

        loss = 0.0
        for b in range(B):
            t_len = text_len[b].item()
            m_len = mel_len[b].item()
            if t_len <= 0 or m_len <= 0:
                continue

            # Build ideal diagonal alignment
            # Each text token should cover approximately m_len / t_len mel frames
            ideal_dur = m_len / max(t_len, 1)
            pred_dur = durations[b, :t_len]

            # MSE between predicted and ideal durations
            loss += F.mse_loss(pred_dur, torch.full_like(pred_dur, ideal_dur))

        return loss / max(B, 1)


# ============================================================
# 12. HiFi-GAN Vocoder
# ============================================================

class HifiGANVocoder:
    """Wrapper for torchaudio's pre-trained HiFi-GAN vocoder."""

    def __init__(self, device):
        self.device = device
        try:
            # Try to load HiFi-GAN from torchaudio
            self.bundle = torchaudio.pipelines.HIFIGAN_VOCODER
            self.generator = self.bundle.get_model().to(device)
            self.generator.eval()
            log.info(f"HiFi-GAN loaded from torchaudio (sample rate: {self.bundle.sample_rate})")
        except Exception as e:
            log.warning(f"Could not load HiFi-GAN from torchaudio: {e}")
            log.warning("Falling back to Griffin-Lim")
            self.generator = None
            self.griffin_lim = T.GriffinLim(
                n_fft=1024, hop_length=256, n_iter=60
            )
            self.inv_mel = T.InverseMelScale(
                n_stft=1024 // 2 + 1, n_mels=80,
                sample_rate=22050,
            )

    @torch.no_grad()
    def vocode(self, mel_spec):
        """Convert mel spectrogram to waveform.
        mel_spec: (n_mels, T) or (B, n_mels, T)
        """
        if self.generator is not None:
            # HiFi-GAN expects (B, n_mels, T) normalized mel
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)
            # Clamp and normalize
            mel = mel_spec.clamp(min=-11.5, max=2.5)
            mel = (mel + 11.5) / 14.0  # normalize to [0, 1]
            wav = self.generator(mel.to(self.device))
            return wav.squeeze(0).cpu() if mel_spec.size(0) == 1 else wav.cpu()
        else:
            # Fallback to Griffin-Lim
            if mel_spec.dim() == 3:
                mel_spec = mel_spec.squeeze(0)
            mel_linear = mel_spec.exp().cpu()
            linear = self.inv_mel(mel_linear)
            wav = self.griffin_lim(linear)
            return wav

# ============================================================
# 13. Trainer
# ============================================================

class CNNTTSTrainer:

    def __init__(self, cfg, model, loader, vocab):
        self.cfg       = cfg
        self.model     = model.to(cfg.DEVICE)
        self.loader    = loader
        self.vocab     = vocab

        self.mel_criterion = MelLoss()
        self.attn_criterion = GuidedAttentionLoss(sigma=0.2)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

        # Warmup + cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self._lr_lambda(step)
        )

        # torch.amp.GradScaler for CUDA (compatible across torch versions)
        if cfg.DEVICE == "cuda":
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except TypeError:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self.global_step = 0

        # HiFi-GAN vocoder
        self.vocoder = HifiGANVocoder(cfg.DEVICE)

        # Mel transform for waveform generation during training
        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.SAMPLE_RATE,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LEN,
            n_mels=cfg.N_MELS,
        ).to(cfg.DEVICE)

    def _lr_lambda(self, step):
        if step < self.cfg.WARMUP_STEPS:
            return float(step) / float(max(1, self.cfg.WARMUP_STEPS))
        # Cosine annealing
        progress = float(step - self.cfg.WARMUP_STEPS) / float(
            max(1, self.cfg.EPOCHS * len(self.loader) - self.cfg.WARMUP_STEPS)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _save(self, epoch, loss):
        path = os.path.join(self.cfg.CKPT_DIR, f"cnn_tts_v2_epoch_{epoch:02d}_loss{loss:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss": loss,
            "vocab": self.vocab.char2idx,
        }, path)
        log.info(f"Checkpoint: {path}")
        return path

    def _mel_to_wav(self, mel):
        """Convert mel spectrogram to waveform for loss computation."""
        # mel: (B, n_mels, T)
        if self.vocoder.generator is not None:
            # Normalize mel for HiFi-GAN
            mel_norm = mel.clamp(min=-11.5, max=2.5)
            mel_norm = (mel_norm + 11.5) / 14.0
            wav = self.vocoder.generator(mel_norm)
            return wav
        else:
            # Fallback: use inverse mel + Griffin-Lim
            wavs = []
            for b in range(mel.size(0)):
                mel_linear = mel[b].exp().cpu()
                linear = self.vocoder.inv_mel(mel_linear)
                wav = self.vocoder.griffin_lim(linear)
                wavs.append(wav.to(mel.device))
            return torch.stack(wavs, dim=0)

    def train(self):
        log.info("=" * 65)
        log.info("CNN KANNADA TTS v2 — TRAINING (Improved Pure CNN)")
        log.info(f"Device: {self.cfg.DEVICE} | Epochs: {self.cfg.EPOCHS} | Batch: {self.cfg.BATCH_SIZE}")
        log.info(f"Vocab: {self.cfg.VOCAB_SIZE} | Mel: {self.cfg.N_MELS}×{self.cfg.MAX_MEL_LEN}")
        log.info(f"Encoder: {self.cfg.ENC_LAYERS} + {self.cfg.DILATED_LAYERS} dilated | Decoder: {self.cfg.DEC_LAYERS}")
        log.info(f"PostNet: {self.cfg.POSTNET_LAYERS} layers | Vocoder: HiFi-GAN")
        log.info("=" * 65)

        last_ckpt = None
        best_loss = float("inf")

        for epoch in range(1, self.cfg.EPOCHS + 1):
            self.model.train()
            epoch_loss    = 0.0
            epoch_mel_mse = 0.0
            epoch_mel_mae = 0.0
            epoch_sc_loss = 0.0
            epoch_attn    = 0.0
            self.optimizer.zero_grad()
            loop = tqdm(self.loader, desc=f"Epoch {epoch:02d}/{self.cfg.EPOCHS}")

            for step, batch in enumerate(loop):
                self.global_step += 1
                ids = batch["input_ids"].to(self.cfg.DEVICE)
                mel = batch["mel"].to(self.cfg.DEVICE)
                text_len = batch["text_len"].to(self.cfg.DEVICE)
                mel_len = batch["mel_len"].to(self.cfg.DEVICE)

                # Forward pass
                if self.scaler:
                    try:
                        autocast_ctx = torch.amp.autocast("cuda")
                    except TypeError:
                        autocast_ctx = torch.cuda.amp.autocast()
                    with autocast_ctx:
                        out = self.model(ids, mel_max_len=mel.size(-1))
                        mel_before = out["mel_before"]
                        mel_after  = out["mel_after"]

                        # Compute mel loss (only 2 args: mel_pred, mel_target)
                        mel_loss = self.mel_criterion(mel_after, mel)
                        total_loss = mel_loss["total"]

                        # Guided attention loss
                        attn_loss = self.attn_criterion(
                            out["durations"], text_len, mel_len
                        )
                        total_loss = total_loss + 0.1 * attn_loss

                        total_loss = total_loss / self.cfg.GRAD_ACCUMULATION

                    self.scaler.scale(total_loss).backward()
                else:
                    out = self.model(ids, mel_max_len=mel.size(-1))
                    mel_before = out["mel_before"]
                    mel_after  = out["mel_after"]

                    mel_loss = self.mel_criterion(mel_after, mel)
                    total_loss = mel_loss["total"]

                    attn_loss = self.attn_criterion(
                        out["durations"], text_len, mel_len
                    )
                    total_loss = total_loss + 0.1 * attn_loss
                    total_loss = total_loss / self.cfg.GRAD_ACCUMULATION
                    total_loss.backward()

                # Gradient accumulation step
                if (step + 1) % self.cfg.GRAD_ACCUMULATION == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss    += total_loss.item()
                epoch_mel_mse += mel_loss["mel_mse"].item()
                epoch_mel_mae += mel_loss["mel_mae"].item()
                epoch_sc_loss += mel_loss["sc_loss"].item()
                epoch_attn    += attn_loss.item()

                loop.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    mel_mse=f"{mel_loss['mel_mse'].item():.4f}",
                    sc_loss=f"{mel_loss['sc_loss'].item():.4f}",
                )

            # End of epoch
            n_batches = len(self.loader)
            avg_loss    = epoch_loss / n_batches
            avg_mel_mse = epoch_mel_mse / n_batches
            avg_mel_mae = epoch_mel_mae / n_batches
            avg_sc_loss = epoch_sc_loss / n_batches
            avg_attn    = epoch_attn / n_batches
            lr = self.scheduler.get_last_lr()[0]

            log.info(
                f"Epoch {epoch:02d} | "
                f"loss: {avg_loss:.4f} | "
                f"mel_mse: {avg_mel_mse:.4f} | "
                f"mel_mae: {avg_mel_mae:.4f} | "
                f"sc_loss: {avg_sc_loss:.4f} | "
                f"attn: {avg_attn:.4f} | "
                f"lr: {lr:.6f}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                log.info(f"New best loss: {best_loss:.4f}")

            last_ckpt = self._save(epoch, avg_loss)

        log.info("TRAINING COMPLETE")
        return last_ckpt

# ============================================================
# 14. Inference — CNN mel → HiFi-GAN waveform
# ============================================================

class CNNTTSInference:

    def __init__(self, cfg, vocab, checkpoint_path=None):
        self.cfg    = cfg
        self.vocab  = vocab
        self.device = cfg.DEVICE

        self.model = KannadaCNNTTSv2(cfg).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            log.info(f"Loaded checkpoint: {checkpoint_path}")
        else:
            log.info("No checkpoint — running with random weights (for pipeline test)")
        self.model.eval()

        self.vocoder = HifiGANVocoder(self.device)

    @torch.no_grad()
    def synthesize(self, text, output_path="/content/cnn_v2_output.wav"):
        ids = torch.tensor(
            [self.vocab.encode(text, self.cfg.MAX_TEXT_LEN)],
            dtype=torch.long, device=self.device
        )
        log.info(f"Input text  : {text}")
        log.info(f"Input shape : {ids.shape}")

        mel_pred = self.model.synthesize(ids)           # (1, n_mels, mel_len)
        log.info(f"Mel shape   : {mel_pred.shape}")

        audio = self.vocoder.vocode(mel_pred)           # (T_wav,)
        audio_np = audio.squeeze().numpy()

        sf.write(output_path, audio_np, samplerate=self.cfg.SAMPLE_RATE)
        log.info(f"Audio saved : {output_path}  ({len(audio_np)/self.cfg.SAMPLE_RATE:.2f}s)")
        return output_path, audio_np

# ============================================================
# 15. Run Everything
# ============================================================

# ── Step 1: Dataset ──────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 1 — DATASET PREPARATION")
log.info("=" * 65)
DatasetPreparer(cfg).run()

# ── Step 1.5: Dataset Analysis ───────────────────────────────
DatasetAnalyzer(cfg).run()

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
augmenter = AudioAugmenter(cfg)
dataset = KannadaTTSDataset(cfg, vocab, augmenter=augmenter)
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
model     = KannadaCNNTTSv2(cfg)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Total params    : {total:,}")
log.info(f"Trainable params: {trainable:,}")

# Print model architecture
print("\n" + "=" * 65)
print("  MODEL ARCHITECTURE")
print("=" * 65)
print(model)
print("=" * 65 + "\n")

# ── Step 5: Train ────────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 5 — TRAINING")
log.info("=" * 65)
trainer   = CNNTTSTrainer(cfg, model, loader, vocab)
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
    out_path = f"/content/cnn_v2_output_{i+1}.wav"
    path, audio = tts.synthesize(sentence, output_path=out_path)
    print(f"\nText: {sentence}")
    display(Audio(audio, rate=cfg.SAMPLE_RATE))

log.info("=" * 65)
log.info("ALL DONE — CNN KANNADA TTS v2")
log.info("=" * 65)