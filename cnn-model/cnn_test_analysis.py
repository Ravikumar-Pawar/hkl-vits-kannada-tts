# ============================================================
# CNN Kannada TTS v2 — Test Analysis (Updated for v2 checkpoint)
# Pure CNN only — NO MMS / NO VITS
# Single Colab Cell — Copy & Paste and Run
# Loads best v2 checkpoint, tests 100 dataset samples, all metrics
# ============================================================

# ============================================================
# 1. Install
# ============================================================

!pip install -q torch torchaudio librosa soundfile pandas requests tqdm numpy transformers pesq pystoi jiwer scipy matplotlib

# ============================================================
# 2. Imports
# ============================================================

import os, zipfile, json, logging, math, glob
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
from tqdm import tqdm
from IPython.display import Audio, display

# ============================================================
# 3. Logger
# ============================================================

def setup_logger(name="cnn_test"):
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch     :", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("Device    :", DEVICE, f"({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")

# ============================================================
# 5. Config — must match colab_single_cell.py v2 exactly
# ============================================================

class Config:
    BASE        = "/content"
    DATA_DIR    = BASE + "/kannada_tts_dataset"
    WAV_DIR     = DATA_DIR + "/wavs"
    TEXT_DIR    = DATA_DIR + "/text"
    CKPT_DIR    = BASE + "/cnn_tts_v2_checkpoints"   # v2 checkpoint dir
    VOCAB_PATH  = BASE + "/cnn_tts_v2_vocab.json"     # v2 vocab path
    OUT_DIR     = BASE + "/cnn_test_outputs"

    WAV_ZIP_URL = "https://openslr.trmal.net/resources/79/kn_in_male.zip"
    TSV_URL     = "https://openslr.trmal.net/resources/79/line_index_male.tsv"
    ZIP_NAME    = "dataset.zip"
    TSV_NAME    = "metadata.tsv"

    # Audio — must match training
    SAMPLE_RATE  = 22050
    N_FFT        = 1024
    HOP_LEN      = 256
    N_MELS       = 80
    MAX_WAV_LEN  = 220500
    MAX_MEL_LEN  = 860
    MAX_TEXT_LEN = 150

    # CNN Model — must match training (v2)
    VOCAB_SIZE     = None   # filled after vocab load
    EMBED_DIM      = 256
    ENC_LAYERS     = 4       # standard DSConv blocks
    DILATED_LAYERS = 4       # dilated DSConv blocks
    DEC_LAYERS     = 4       # mel refinement blocks
    POSTNET_LAYERS = 5       # post-net refinement
    KERNEL_SIZE    = 5
    DROPOUT        = 0.1

    # Duration predictor
    DURATION_HIDDEN = 128

    NUM_TEST_SAMPLES = 100
    DEVICE = DEVICE

cfg = Config()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

# ============================================================
# 6. Dataset Preparation
# ============================================================

def prepare_dataset(cfg):
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.WAV_DIR,  exist_ok=True)

    zip_path = os.path.join(cfg.DATA_DIR, cfg.ZIP_NAME)
    tsv_path = os.path.join(cfg.DATA_DIR, cfg.TSV_NAME)

    def _download(url, path):
        log.info(f"Downloading {url}")
        r = requests.get(url, stream=True, timeout=60)
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f:
            for chunk in tqdm(r.iter_content(1024), total=total // 1024, unit="KB"):
                f.write(chunk)

    if not os.path.exists(zip_path):
        _download(cfg.WAV_ZIP_URL, zip_path)
    else:
        log.info("ZIP exists — skip")

    if not os.path.exists(tsv_path):
        _download(cfg.TSV_URL, tsv_path)
    else:
        log.info("TSV exists — skip")

    wav_files = [f for f in os.listdir(cfg.WAV_DIR) if f.endswith(".wav")]
    if not wav_files:
        log.info("Extracting WAVs...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cfg.WAV_DIR)
        log.info("Extraction done")
    else:
        log.info(f"WAVs ready: {len(wav_files)} files")

    if not os.path.exists(cfg.TEXT_DIR):
        os.makedirs(cfg.TEXT_DIR, exist_ok=True)
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["wav_file", "text"])
        for _, row in df.iterrows():
            p = os.path.join(cfg.TEXT_DIR, str(row["wav_file"]).strip() + ".txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(str(row["text"]).strip())
        log.info(f"Text files written: {len(df)}")
    else:
        log.info("Text files ready — skip")

# ============================================================
# 7. Dataset Analysis — uses librosa.get_duration (fixes 0.00s issue)
# ============================================================

class DatasetAnalyzer:
    """Analyzes TTS dataset distribution: text lengths, audio durations, character stats."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _load_metadata(self):
        """Load text files and corresponding WAV durations using librosa (reliable)."""
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
            # Use librosa.get_duration — more reliable than sf.SoundFile
            try:
                duration = librosa.get_duration(filename=wav_path, sr=self.cfg.SAMPLE_RATE)
            except Exception:
                try:
                    info = sf.SoundFile(wav_path)
                    duration = info.frames / info.samplerate
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
        all_chars = "".join(df["text"].values)
        char_counts = {}
        for c in all_chars:
            char_counts[c] = char_counts.get(c, 0) + 1
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])

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
# 8. Vocabulary — identical to colab_single_cell.py
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
# 9. Model — v2 architecture (identical to colab_single_cell.py)
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
    """Improved Pure CNN Text-to-Mel: CNN encoder + duration predictor + length regulator + CNN decoder + PostNet."""

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
        """Inference: encode → predict duration → regulate → decode → postnet."""
        self.eval()
        enc = self.encoder(input_ids)
        durations = self.duration_predictor(enc)
        reg = self.length_regulator(enc, durations)
        mel_before = self.decoder(reg)
        mel_after = self.postnet(mel_before)
        return mel_after

# ============================================================
# 10. HifiGAN Vocoder (with Griffin-Lim fallback)
# ============================================================

class HifiGANVocoder:
    """Wrapper for torchaudio's pre-trained HiFi-GAN vocoder with Griffin-Lim fallback."""

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
# 11. Best Checkpoint Loader — v2 naming convention
# ============================================================

def load_best_checkpoint(cfg, model):
    """Picks checkpoint with lowest loss from filename.
    v2 naming: cnn_tts_v2_epoch_XX_lossY.YYYY.pt
    """
    pts = glob.glob(os.path.join(cfg.CKPT_DIR, "cnn_tts_v2_epoch_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No v2 checkpoints found in {cfg.CKPT_DIR}")

    # Parse loss from filename: cnn_tts_v2_epoch_XX_lossY.YYYY.pt
    def parse_loss(path):
        try:
            return float(os.path.basename(path).split("_loss")[1].replace(".pt", ""))
        except Exception:
            return float("inf")

    best_path = min(pts, key=parse_loss)
    ckpt      = torch.load(best_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model"])
    log.info(f"Best checkpoint : {best_path}")
    log.info(f"Epoch           : {ckpt.get('epoch', '?')}  |  Loss: {ckpt.get('loss', '?'):.4f}")
    return best_path, ckpt.get("epoch", "?"), ckpt.get("loss", float("nan"))

# ============================================================
# 12. Inference — v2 model.synthesize + HifiGANVocoder
# ============================================================

@torch.no_grad()
def synthesize(text, model, vocab, cfg, vocoder):
    """Synthesize speech from text using v2 model + vocoder."""
    ids = torch.tensor(
        [vocab.encode(text, cfg.MAX_TEXT_LEN)],
        dtype=torch.long, device=cfg.DEVICE
    )
    mel_pred = model.synthesize(ids)                     # (1, n_mels, mel_len) — log-mel
    audio    = vocoder.vocode(mel_pred)                  # (T_wav,)
    audio_np = audio.squeeze().numpy()
    return audio_np

# ============================================================
# 13. Metrics
# ============================================================

def mel_cepstral_distortion(ref, syn, sr):
    r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=sr, n_mfcc=13)
    s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=sr, n_mfcc=13)
    n = min(r.shape[1], s.shape[1])
    d = r[:, :n] - s[:, :n]
    return float((10.0 / np.log(10)) * np.sqrt(2.0 * np.sum(d ** 2, axis=0)).mean())

def log_spectral_distance(ref, syn, sr, n_fft=1024, hop=256):
    r = np.abs(librosa.stft(ref.astype(np.float32), n_fft=n_fft, hop_length=hop)) + 1e-8
    s = np.abs(librosa.stft(syn.astype(np.float32), n_fft=n_fft, hop_length=hop)) + 1e-8
    n = min(r.shape[1], s.shape[1])
    return float(np.sqrt(np.mean((np.log(r[:, :n]) - np.log(s[:, :n])) ** 2)))

def f0_rmse(ref, syn, sr):
    rf = librosa.yin(ref.astype(np.float32), fmin=50, fmax=500, sr=sr)
    sf_ = librosa.yin(syn.astype(np.float32), fmin=50, fmax=500, sr=sr)
    n   = min(len(rf), len(sf_))
    v   = (rf[:n] > 0) & (sf_[:n] > 0)
    return float(np.sqrt(np.mean((rf[:n][v] - sf_[:n][v]) ** 2))) if v.sum() > 0 else float("nan")

def energy_rmse(ref, syn, hop=256):
    r = librosa.feature.rms(y=ref.astype(np.float32), hop_length=hop)[0]
    s = librosa.feature.rms(y=syn.astype(np.float32), hop_length=hop)[0]
    n = min(len(r), len(s))
    return float(np.sqrt(np.mean((r[:n] - s[:n]) ** 2)))

def duration_ratio(ref, syn):
    return float(len(syn) / max(len(ref), 1))

def snr_db(ref, syn):
    n   = min(len(ref), len(syn))
    r   = ref[:n].astype(np.float64)
    s   = syn[:n].astype(np.float64)
    sig = np.mean(r ** 2)
    nse = np.mean((r - s) ** 2) + 1e-10
    return float(10 * np.log10(sig / nse))

def pesq_score(ref, syn, sr):
    try:
        from pesq import pesq
        tsr = 16000
        if sr != tsr:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=sr, target_sr=tsr)
            syn = librosa.resample(syn.astype(np.float32), orig_sr=sr, target_sr=tsr)
        n = min(len(ref), len(syn))
        return float(pesq(tsr, ref[:n], syn[:n], "wb"))
    except Exception:
        return float("nan")

def stoi_score(ref, syn, sr):
    try:
        from pystoi import stoi
        n = min(len(ref), len(syn))
        return float(stoi(ref[:n], syn[:n], sr, extended=False))
    except Exception:
        return float("nan")

def compute_all_metrics(ref, syn, sr):
    return {
        "MCD (dB)"       : mel_cepstral_distortion(ref, syn, sr),
        "LSD"            : log_spectral_distance(ref, syn, sr),
        "F0 RMSE (Hz)"   : f0_rmse(ref, syn, sr),
        "Energy RMSE"    : energy_rmse(ref, syn),
        "Duration Ratio" : duration_ratio(ref, syn),
        "SNR (dB)"       : snr_db(ref, syn),
        "PESQ"           : pesq_score(ref, syn, sr),
        "STOI"           : stoi_score(ref, syn, sr),
    }

# ============================================================
# 14. Run
# ============================================================

# ── Step 1: Dataset ──────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 1 — DATASET PREPARATION")
log.info("=" * 65)
prepare_dataset(cfg)

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

# ── Step 3: Build model & load best v2 checkpoint ──────────────
log.info("=" * 65)
log.info("STEP 3 — MODEL & BEST CHECKPOINT")
log.info("=" * 65)
model = KannadaCNNTTSv2(cfg).to(cfg.DEVICE)
model.eval()
ckpt_path, ckpt_epoch, ckpt_loss = load_best_checkpoint(cfg, model)

total     = sum(p.numel() for p in model.parameters())
log.info(f"Model params: {total:,}")

# ── Step 4: Collect 100 test samples from dataset ──────────────
log.info("=" * 65)
log.info("STEP 4 — COLLECTING 100 TEST SAMPLES")
log.info("=" * 65)
items = []
for fname in sorted(os.listdir(cfg.TEXT_DIR)):
    if not fname.endswith(".txt"):
        continue
    wav_path = os.path.join(cfg.WAV_DIR, fname.replace(".txt", ".wav"))
    if not os.path.exists(wav_path):
        continue
    with open(os.path.join(cfg.TEXT_DIR, fname), encoding="utf-8") as f:
        text = f.read().strip()
    if text:
        items.append({"text": text, "wav": wav_path})

np.random.seed(42)
indices      = np.random.choice(len(items), size=min(cfg.NUM_TEST_SAMPLES, len(items)), replace=False)
test_samples = [items[i] for i in indices]
log.info(f"Selected {len(test_samples)} / {len(items)} samples")

# ── Step 5: Synthesize & Evaluate ────────────────────────────
log.info("=" * 65)
log.info("STEP 5 — SYNTHESIS & METRIC COMPUTATION")
log.info("=" * 65)

# Initialize vocoder (HiFi-GAN with Griffin-Lim fallback)
vocoder = HifiGANVocoder(cfg.DEVICE)

results = []

for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
    text     = sample["text"]
    wav_path = sample["wav"]

    # Reference audio from dataset (resampled to match model SR)
    ref_audio, _ = librosa.load(wav_path, sr=cfg.SAMPLE_RATE, mono=True)

    try:
        syn_audio = synthesize(text, model, vocab, cfg, vocoder)
    except Exception as e:
        log.warning(f"[{idx}] Synthesis failed: {e}")
        continue

    out_path = os.path.join(cfg.OUT_DIR, f"syn_{idx:03d}.wav")
    sf.write(out_path, syn_audio, samplerate=cfg.SAMPLE_RATE)

    metrics = compute_all_metrics(ref_audio, syn_audio, cfg.SAMPLE_RATE)
    metrics["sample_idx"] = idx
    metrics["text"]       = text
    results.append(metrics)

# ── Step 6: Summary ──────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 6 — RESULTS SUMMARY")
log.info("=" * 65)

df          = pd.DataFrame(results)
metric_cols = ["MCD (dB)", "LSD", "F0 RMSE (Hz)", "Energy RMSE",
               "Duration Ratio", "SNR (dB)", "PESQ", "STOI"]
summary     = df[metric_cols].agg(["mean", "std", "min", "max"]).T
summary.columns = ["Mean", "Std", "Min", "Max"]

print("\n" + "=" * 65)
print("  CNN KANNADA TTS v2 — TEST ANALYSIS  (Pure CNN, No MMS/VITS)")
print("=" * 65)
print(f"  Checkpoint : {os.path.basename(ckpt_path)}")
print(f"  Epoch      : {ckpt_epoch}  |  Train Loss: {ckpt_loss:.4f}")
print(f"  Vocoder    : {'HiFi-GAN' if vocoder.generator is not None else 'Griffin-Lim'}")
print(f"  Samples    : {len(results)} / {cfg.NUM_TEST_SAMPLES}")
print("=" * 65)
print(summary.round(4).to_string())
print("=" * 65)
print("""
METRIC GUIDE:
  MCD (dB)       — Mel Cepstral Distortion   | lower is better | good < 8 dB
  LSD            — Log-Spectral Distance      | lower is better
  F0 RMSE (Hz)   — Pitch accuracy             | lower is better
  Energy RMSE    — Loudness accuracy          | lower is better
  Duration Ratio — Syn/Ref length ratio       | 1.0 = perfect
  SNR (dB)       — Signal-to-Noise Ratio      | higher is better
  PESQ           — Perceptual quality         | higher is better | -0.5 to 4.5
  STOI           — Speech intelligibility     | higher is better | 0 to 1
""")

csv_path = os.path.join(cfg.OUT_DIR, "cnn_test_results.csv")
df.to_csv(csv_path, index=False)
log.info(f"Full results saved: {csv_path}")

# ── Step 7: Sample Playback ───────────────────────────────────
log.info("=" * 65)
log.info("STEP 7 — SAMPLE AUDIO PLAYBACK (first 3)")
log.info("=" * 65)

for i in range(min(3, len(results))):
    r        = results[i]
    out_path = os.path.join(cfg.OUT_DIR, f"syn_{r['sample_idx']:03d}.wav")
    audio, _ = librosa.load(out_path, sr=cfg.SAMPLE_RATE)
    print(f"\n[{i+1}] Text : {r['text']}")
    print(f"     MCD  : {r['MCD (dB)']:.2f} dB | PESQ: {r['PESQ']:.3f} | STOI: {r['STOI']:.3f} | SNR: {r['SNR (dB)']:.2f} dB")
    display(Audio(audio, rate=cfg.SAMPLE_RATE))

log.info("=" * 65)
log.info("CNN TEST ANALYSIS COMPLETE")
log.info("=" * 65)
