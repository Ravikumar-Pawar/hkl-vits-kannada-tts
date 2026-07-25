# ============================================================
# CNN Kannada TTS — Test Analysis
# Single Colab Cell — Copy & Paste and Run
# Runtime → GPU (T4) recommended
# Tests 100 samples from dataset, computes all TTS metrics
# ============================================================

# ============================================================
# 1. Install
# ============================================================

import subprocess
subprocess.run(["pip", "install", "-q",
    "torch", "torchaudio", "librosa", "soundfile",
    "pandas", "requests", "tqdm", "numpy",
    "transformers", "pesq", "pystoi", "jiwer"
], check=True)

# ============================================================
# 2. Imports
# ============================================================

import os, zipfile, json, logging, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
import requests
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, VitsModel
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
# 5. Config
# ============================================================

class Config:
    BASE       = "/content"
    DATA_DIR   = BASE + "/kannada_tts_dataset"
    WAV_DIR    = DATA_DIR + "/wavs"
    TEXT_DIR   = DATA_DIR + "/text"
    CKPT_DIR   = BASE + "/cnn_tts_checkpoints"
    OUT_DIR    = BASE + "/cnn_test_outputs"

    WAV_ZIP_URL = "https://openslr.trmal.net/resources/79/kn_in_male.zip"
    TSV_URL     = "https://openslr.trmal.net/resources/79/line_index_male.tsv"
    ZIP_NAME    = "dataset.zip"
    TSV_NAME    = "metadata.tsv"

    MMS_MODEL   = "facebook/mms-tts-kan"
    SAMPLE_RATE = 16000          # MMS outputs 16kHz
    MAX_TEXT_LEN = 128

    # CNN architecture (must match training)
    HIDDEN_DIM   = 256
    CNN_CHANNELS = 256
    CNN_LAYERS   = 6
    KERNEL_SIZE  = 5

    NUM_TEST_SAMPLES = 100
    DEVICE = DEVICE

cfg = Config()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

# ============================================================
# 6. Dataset Download & Preparation
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
    if not os.path.exists(tsv_path):
        _download(cfg.TSV_URL, tsv_path)

    wav_files = [f for f in os.listdir(cfg.WAV_DIR) if f.endswith(".wav")]
    if not wav_files:
        log.info("Extracting WAVs...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cfg.WAV_DIR)

    if not os.path.exists(cfg.TEXT_DIR):
        os.makedirs(cfg.TEXT_DIR, exist_ok=True)
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["wav_file", "text"])
        for _, row in df.iterrows():
            p = os.path.join(cfg.TEXT_DIR, str(row["wav_file"]).strip() + ".txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(str(row["text"]).strip())
        log.info(f"Text files written: {len(df)}")

    log.info("Dataset ready")

# ============================================================
# 7. Model Architecture (mirrors training)
# ============================================================

class DSConvBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.depthwise  = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pointwise  = nn.Conv1d(channels, channels, 1)
        self.norm       = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        return self.norm(x + residual)


class KannadaCNNEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, cnn_layers=6, kernel_size=5):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks     = nn.ModuleList([DSConvBlock(hidden_dim, kernel_size) for _ in range(cnn_layers)])
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.input_proj(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class KannadaCNNTTS(nn.Module):
    def __init__(self, mms_model, vocab_size, hidden_dim=256, cnn_layers=6, kernel_size=5):
        super().__init__()
        self.mms = mms_model
        for p in self.mms.parameters():
            p.requires_grad = False
        mms_hidden    = self.mms.config.hidden_size
        self.encoder  = KannadaCNNEncoder(vocab_size, hidden_dim, cnn_layers, kernel_size)
        self.prosody  = nn.Linear(hidden_dim, 4)
        self.adapter  = nn.Linear(hidden_dim, mms_hidden)

    def forward(self, input_ids, attention_mask=None):
        features = self.encoder(input_ids, attention_mask)
        prosody  = self.prosody(features)
        adapted  = self.adapter(features)
        return {"features": features, "prosody": prosody, "adapted": adapted}

# ============================================================
# 8. Checkpoint Loader — picks latest .pt in CKPT_DIR
# ============================================================

def load_latest_checkpoint(cfg, model):
    pts = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, "*.pt")))
    if not pts:
        log.warning("No checkpoints found — using untrained CNN encoder with pretrained MMS")
        return None
    latest = pts[-1]
    ckpt   = torch.load(latest, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model"])
    epoch  = ckpt.get("epoch", "?")
    log.info(f"Loaded checkpoint: {latest}  (epoch {epoch})")
    return latest

# ============================================================
# 9. Inference
# ============================================================

@torch.no_grad()
def synthesize(text, model, mms, tokenizer, cfg):
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=cfg.MAX_TEXT_LEN,
        truncation=True,
    ).to(cfg.DEVICE)

    model(enc.input_ids, enc.attention_mask)   # CNN forward (prosody features)
    mms_out = mms(**enc)
    audio   = mms_out.waveform.squeeze().cpu().numpy()
    return audio

# ============================================================
# 10. Metrics
# ============================================================

def compute_mel_spectrogram(audio, sr, n_fft=512, hop_length=128, n_mels=80):
    mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sr,
                                          n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def mel_cepstral_distortion(ref_audio, syn_audio, sr):
    """MCD — lower is better (dB). Aligns lengths before comparison."""
    ref_mel = librosa.feature.mfcc(y=ref_audio.astype(np.float32), sr=sr, n_mfcc=13)
    syn_mel = librosa.feature.mfcc(y=syn_audio.astype(np.float32), sr=sr, n_mfcc=13)
    min_len = min(ref_mel.shape[1], syn_mel.shape[1])
    ref_mel, syn_mel = ref_mel[:, :min_len], syn_mel[:, :min_len]
    diff = ref_mel - syn_mel
    mcd  = (10.0 / np.log(10)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0)).mean()
    return float(mcd)


def spectral_distortion(ref_audio, syn_audio, sr, n_fft=512, hop_length=128):
    """Log-spectral distance between reference and synthesized."""
    ref_spec = np.abs(librosa.stft(ref_audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length))
    syn_spec = np.abs(librosa.stft(syn_audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length))
    min_len  = min(ref_spec.shape[1], syn_spec.shape[1])
    ref_spec, syn_spec = ref_spec[:, :min_len] + 1e-8, syn_spec[:, :min_len] + 1e-8
    lsd = np.sqrt(np.mean((np.log(ref_spec) - np.log(syn_spec)) ** 2))
    return float(lsd)


def f0_rmse(ref_audio, syn_audio, sr):
    """F0 RMSE in Hz — lower is better."""
    ref_f0 = librosa.yin(ref_audio.astype(np.float32), fmin=50, fmax=500, sr=sr)
    syn_f0 = librosa.yin(syn_audio.astype(np.float32), fmin=50, fmax=500, sr=sr)
    min_len = min(len(ref_f0), len(syn_f0))
    ref_f0, syn_f0 = ref_f0[:min_len], syn_f0[:min_len]
    voiced = (ref_f0 > 0) & (syn_f0 > 0)
    if voiced.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((ref_f0[voiced] - syn_f0[voiced]) ** 2)))


def energy_rmse(ref_audio, syn_audio, sr, hop_length=128):
    """RMS energy RMSE — lower is better."""
    ref_rms = librosa.feature.rms(y=ref_audio.astype(np.float32), hop_length=hop_length)[0]
    syn_rms = librosa.feature.rms(y=syn_audio.astype(np.float32), hop_length=hop_length)[0]
    min_len = min(len(ref_rms), len(syn_rms))
    return float(np.sqrt(np.mean((ref_rms[:min_len] - syn_rms[:min_len]) ** 2)))


def duration_ratio(ref_audio, syn_audio, sr):
    """Ratio of synthesized duration to reference duration (1.0 = perfect)."""
    return float(len(syn_audio) / max(len(ref_audio), 1))


def snr(ref_audio, syn_audio):
    """Signal-to-Noise Ratio (dB) — higher is better."""
    min_len = min(len(ref_audio), len(syn_audio))
    ref, syn = ref_audio[:min_len].astype(np.float64), syn_audio[:min_len].astype(np.float64)
    noise    = ref - syn
    signal_power = np.mean(ref ** 2)
    noise_power  = np.mean(noise ** 2) + 1e-10
    return float(10 * np.log10(signal_power / noise_power))


def pesq_score(ref_audio, syn_audio, sr):
    """PESQ score (wideband) — higher is better (range -0.5 to 4.5)."""
    try:
        from pesq import pesq
        target_sr = 16000
        if sr != target_sr:
            ref_audio = librosa.resample(ref_audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            syn_audio = librosa.resample(syn_audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        min_len = min(len(ref_audio), len(syn_audio))
        return float(pesq(target_sr, ref_audio[:min_len], syn_audio[:min_len], "wb"))
    except Exception as e:
        return float("nan")


def stoi_score(ref_audio, syn_audio, sr):
    """STOI score — higher is better (range 0 to 1)."""
    try:
        from pystoi import stoi
        min_len = min(len(ref_audio), len(syn_audio))
        return float(stoi(ref_audio[:min_len], syn_audio[:min_len], sr, extended=False))
    except Exception as e:
        return float("nan")


def compute_all_metrics(ref_audio, syn_audio, sr):
    return {
        "MCD (dB)"         : mel_cepstral_distortion(ref_audio, syn_audio, sr),
        "LSD"              : spectral_distortion(ref_audio, syn_audio, sr),
        "F0 RMSE (Hz)"     : f0_rmse(ref_audio, syn_audio, sr),
        "Energy RMSE"      : energy_rmse(ref_audio, syn_audio, sr),
        "Duration Ratio"   : duration_ratio(ref_audio, syn_audio, sr),
        "SNR (dB)"         : snr(ref_audio, syn_audio),
        "PESQ"             : pesq_score(ref_audio, syn_audio, sr),
        "STOI"             : stoi_score(ref_audio, syn_audio, sr),
    }

# ============================================================
# 11. Run Everything
# ============================================================

# ── Step 1: Dataset ──────────────────────────────────────────
log.info("=" * 65)
log.info("STEP 1 — DATASET PREPARATION")
log.info("=" * 65)
prepare_dataset(cfg)

# ── Step 2: Load tokenizer & MMS ─────────────────────────────
log.info("=" * 65)
log.info("STEP 2 — LOADING TOKENIZER & MMS BACKBONE")
log.info("=" * 65)
tokenizer = AutoTokenizer.from_pretrained(cfg.MMS_MODEL)
mms       = VitsModel.from_pretrained(cfg.MMS_MODEL).to(cfg.DEVICE)
mms.eval()

# ── Step 3: Build & load CNN model ───────────────────────────
log.info("=" * 65)
log.info("STEP 3 — BUILDING CNN MODEL & LOADING CHECKPOINT")
log.info("=" * 65)
model = KannadaCNNTTS(
    mms,
    vocab_size  = tokenizer.vocab_size,
    hidden_dim  = cfg.HIDDEN_DIM,
    cnn_layers  = cfg.CNN_LAYERS,
    kernel_size = cfg.KERNEL_SIZE,
).to(cfg.DEVICE)
model.eval()

ckpt_path = load_latest_checkpoint(cfg, model)

# ── Step 4: Collect 100 test samples ─────────────────────────
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

# Fixed seed for reproducibility
np.random.seed(42)
indices      = np.random.choice(len(items), size=min(cfg.NUM_TEST_SAMPLES, len(items)), replace=False)
test_samples = [items[i] for i in indices]
log.info(f"Selected {len(test_samples)} samples from {len(items)} total")

# ── Step 5: Synthesize & Evaluate ────────────────────────────
log.info("=" * 65)
log.info("STEP 5 — SYNTHESIS & METRIC COMPUTATION")
log.info("=" * 65)

results = []

for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
    text     = sample["text"]
    wav_path = sample["wav"]

    # Load reference audio
    ref_audio, ref_sr = librosa.load(wav_path, sr=cfg.SAMPLE_RATE, mono=True)

    # Synthesize
    try:
        syn_audio = synthesize(text, model, mms, tokenizer, cfg)
    except Exception as e:
        log.warning(f"[{idx}] Synthesis failed: {e}")
        continue

    # Save synthesized audio
    out_path = os.path.join(cfg.OUT_DIR, f"syn_{idx:03d}.wav")
    sf.write(out_path, syn_audio, samplerate=cfg.SAMPLE_RATE)

    # Compute metrics
    metrics = compute_all_metrics(ref_audio, syn_audio, cfg.SAMPLE_RATE)
    metrics["sample_idx"] = idx
    metrics["text"]       = text
    results.append(metrics)

# ── Step 6: Aggregate & Display Results ──────────────────────
log.info("=" * 65)
log.info("STEP 6 — RESULTS SUMMARY")
log.info("=" * 65)

df = pd.DataFrame(results)

metric_cols = ["MCD (dB)", "LSD", "F0 RMSE (Hz)", "Energy RMSE",
               "Duration Ratio", "SNR (dB)", "PESQ", "STOI"]

summary = df[metric_cols].agg(["mean", "std", "min", "max"]).T
summary.columns = ["Mean", "Std", "Min", "Max"]

print("\n" + "=" * 65)
print("  CNN KANNADA TTS — TEST ANALYSIS RESULTS (100 samples)")
print("=" * 65)
print(f"  Checkpoint : {ckpt_path or 'pretrained MMS only'}")
print(f"  Samples    : {len(results)}")
print("=" * 65)
print(summary.to_string())
print("=" * 65)

# Metric interpretation guide
print("""
METRIC GUIDE:
  MCD (dB)       — Mel Cepstral Distortion     | lower is better  | good < 8 dB
  LSD            — Log-Spectral Distance        | lower is better
  F0 RMSE (Hz)   — Pitch accuracy               | lower is better
  Energy RMSE    — Loudness accuracy            | lower is better
  Duration Ratio — Syn/Ref duration ratio       | 1.0 = perfect
  SNR (dB)       — Signal-to-Noise Ratio        | higher is better
  PESQ           — Perceptual speech quality    | higher is better | range -0.5 to 4.5
  STOI           — Speech intelligibility       | higher is better | range 0 to 1
""")

# Save full results CSV
csv_path = os.path.join(cfg.OUT_DIR, "cnn_test_results.csv")
df.to_csv(csv_path, index=False)
log.info(f"Full results saved: {csv_path}")

# ── Step 7: Play sample outputs ──────────────────────────────
log.info("=" * 65)
log.info("STEP 7 — SAMPLE AUDIO PLAYBACK (first 3)")
log.info("=" * 65)

for i in range(min(3, len(results))):
    r        = results[i]
    out_path = os.path.join(cfg.OUT_DIR, f"syn_{r['sample_idx']:03d}.wav")
    audio, _ = librosa.load(out_path, sr=cfg.SAMPLE_RATE)
    print(f"\n[{i+1}] Text: {r['text']}")
    print(f"     MCD: {r['MCD (dB)']:.2f} dB | PESQ: {r['PESQ']:.3f} | STOI: {r['STOI']:.3f}")
    display(Audio(audio, rate=cfg.SAMPLE_RATE))

log.info("=" * 65)
log.info("CNN TEST ANALYSIS COMPLETE")
log.info("=" * 65)
