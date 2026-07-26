# CNN Kannada TTS — Implementation Documentation

**File:** `cnn-model/colab_single_cell.py`
**Runtime:** Google Colab, GPU (T4 recommended)

---

## Overview

`colab_single_cell.py` is a self-contained, single-cell Google Colab script that implements a complete **Pure CNN Text-to-Speech pipeline for Kannada**. It covers the full lifecycle from dataset download to trained model inference — without any dependency on MMS, VITS, or any pretrained speech backbone.

---

## Pipeline Flow

```
Install Dependencies
        │
        ▼
Dataset Download & Preparation
  (OpenSLR kn_in_male WAVs + TSV)
        │
        ▼
Dataset Analysis
  (text length, duration, char frequency, scatter plots)
        │
        ▼
Character Vocabulary Build / Load
  (KannadaVocab — char-level, 107 tokens)
        │
        ▼
PyTorch Dataset + DataLoader
  (mel spectrogram extraction, augmentation)
        │
        ▼
Model Construction — KannadaCNNTTS
  ┌─────────────────────────────────────┐
  │  CNNTextEncoder                     │
  │    Embedding → PositionalEncoding   │
  │    → Standard DSConv × 4            │
  │    → Dilated DSConv × 4             │
  │                                     │
  │  DurationPredictor                  │
  │    Conv1d × 2 → Linear → Softplus   │
  │                                     │
  │  LengthRegulator                    │
  │    Token expansion by duration      │
  │                                     │
  │  CNNMelDecoder                      │
  │    DSConv × 4 → Linear → mel        │
  │                                     │
  │  PostNet                            │
  │    Conv1d × 5 → residual add        │
  └─────────────────────────────────────┘
        │
        ▼
Training Loop
  (MelLoss + GuidedAttentionLoss,
   mixed precision, grad accumulation,
   warmup + cosine LR, checkpoint saving)
        │
        ▼
Inference
  (text → mel → HiFi-GAN → waveform → .wav)
```

---

## Section-by-Section Documentation

### Section 1 — Install

```python
!pip install -q torch torchaudio librosa soundfile pandas requests tqdm numpy transformers pesq pystoi jiwer
```

Installs all required packages in the Colab environment. `pesq` and `pystoi` are included for downstream evaluation use in `cnn_test_analysis.py`.

---

### Section 2 — Imports

Key imports and their roles:

| Import | Purpose |
| --- | --- |
| `torch`, `torch.nn` | Model definition, training |
| `torchaudio`, `torchaudio.transforms` | Mel spectrogram, Griffin-Lim, HiFi-GAN |
| `librosa` | Audio loading, augmentation, analysis |
| `soundfile` | Writing output WAV files |
| `matplotlib` | Dataset analysis plots |
| `matplotlib.font_manager` | Kannada font detection for plot labels |

**Kannada font handling:** The script scans installed fonts for Noto/Kannada/Lohit families and sets them as the preferred matplotlib font so Kannada characters render correctly in plots. If none are found, it falls back to Unicode codepoints.

---

### Section 3 — Logger

```python
def setup_logger(name="cnn_tts_v2")
```

Creates a named Python logger with timestamp formatting. Uses `if not logger.handlers` guard to prevent duplicate handlers when the cell is re-run in the same Colab session.

---

### Section 4 — GPU Check

Detects CUDA availability and sets the global `DEVICE` variable (`"cuda"` or `"cpu"`). Prints torch and torchaudio versions for reproducibility tracking.

---

### Section 5 — Config

```python
class Config
```

Central configuration class. All hyperparameters and paths are defined here. Key groups:

**Paths**

| Field | Value | Description |
| --- | --- | --- |
| `BASE` | `/content` | Colab root |
| `DATA_DIR` | `/content/kannada_tts_dataset` | Dataset root |
| `WAV_DIR` | `DATA_DIR/wavs` | WAV audio files |
| `TEXT_DIR` | `DATA_DIR/text` | Per-utterance text files |
| `CKPT_DIR` | `/content/cnn_tts_v2_checkpoints` | Saved checkpoints |
| `VOCAB_PATH` | `/content/cnn_tts_v2_vocab.json` | Saved vocabulary |

**Audio**

| Field | Value | Description |
| --- | --- | --- |
| `SAMPLE_RATE` | 22050 | Target sample rate (Hz) |
| `N_FFT` | 1024 | FFT window size |
| `HOP_LEN` | 256 | Hop length for STFT |
| `N_MELS` | 80 | Mel filter banks |
| `MAX_WAV_LEN` | 220500 | Max waveform samples (10s) |
| `MAX_MEL_LEN` | 860 | Max mel frames |

**Model Architecture**

| Field | Value | Description |
| --- | --- | --- |
| `VOCAB_SIZE` | None → set at runtime | Character vocabulary size |
| `EMBED_DIM` | 256 | Embedding and hidden dimension |
| `ENC_LAYERS` | 4 | Standard DSConv encoder blocks |
| `DILATED_LAYERS` | 4 | Dilated DSConv encoder blocks |
| `DEC_LAYERS` | 4 | Mel decoder DSConv blocks |
| `POSTNET_LAYERS` | 5 | PostNet Conv1d layers |
| `KERNEL_SIZE` | 5 | Conv kernel size throughout |
| `DROPOUT` | 0.1 | Dropout rate |
| `DURATION_HIDDEN` | 128 | Duration predictor hidden dim |

**Training**

| Field | Value | Description |
| --- | --- | --- |
| `EPOCHS` | 30 | Total training epochs |
| `BATCH_SIZE` | 8 | Samples per batch |
| `LEARNING_RATE` | 1e-3 | Peak learning rate |
| `GRAD_ACCUMULATION` | 4 | Effective batch = 8 × 4 = 32 |
| `GRAD_CLIP` | 1.0 | Gradient norm clipping |
| `WARMUP_STEPS` | 1000 | LR warmup steps |
| `SPEED_PERTURB` | True | Enable speed augmentation |
| `NOISE_STD` | 0.005 | Gaussian noise std for augmentation |

---

### Section 6 — DatasetPreparer

```python
class DatasetPreparer
```

Downloads and prepares the OpenSLR Kannada dataset (resource 79, `kn_in_male`).

**Steps:**

1. Downloads `kn_in_male.zip` (WAV files) if not present
2. Downloads `line_index_male.tsv` (text metadata) if not present
3. Extracts WAV files to `WAV_DIR`
4. Parses the TSV and writes one `.txt` file per utterance into `TEXT_DIR`

All steps are idempotent — skipped if files already exist.

**Dataset source:** OpenSLR resource 79 (`kn_in_male`) — male Kannada speaker, ~2214 utterances.

---

### Section 7 — DatasetAnalyzer

```python
class DatasetAnalyzer
```

Performs statistical analysis of the dataset before training. Produces a 2×2 matplotlib figure and logs key statistics.

**Plots generated:**

| Plot | Description |
| --- | --- |
| Text Length Distribution | Histogram of character counts per utterance with median and P95 lines |
| Audio Duration Distribution | Histogram of WAV durations in seconds, shows total dataset hours |
| Top 25 Character Frequency | Horizontal bar chart of most frequent Kannada characters |
| Text Length vs Duration | Scatter plot with linear regression line showing text-duration correlation |

**Logged statistics:**

- Text length: mean, std, min, max, P50, P95
- Audio duration: mean, std, min, max, total hours
- Mel frame estimates: mean, max, P95, percentage truncated by `MAX_MEL_LEN`
- Unique character count

**Return value:** Dictionary with `n_samples`, `avg_text_len`, `avg_duration`, `total_hours`, `unique_chars`.

---

### Section 8 — KannadaVocab

```python
class KannadaVocab
```

Character-level vocabulary for Kannada text encoding.

**Special tokens:**

| Token | Index | Purpose |
| --- | --- | --- |
| `<pad>` | 0 | Padding |
| `<bos>` | 1 | Begin of sequence |
| `<eos>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown character |

**Methods:**

- `build(text_dir, save_path)` — Scans all `.txt` files, collects unique characters, builds sorted vocabulary, saves to JSON. Returns vocab size.
- `load(path)` — Loads vocabulary from JSON. Returns vocab size.
- `encode(text, max_len)` — Converts text string to padded integer list: `[BOS, char_ids..., EOS, PAD...]`. Truncates at `max_len`.
- `decode(ids)` — Converts integer list back to string, skipping special tokens.

**Typical vocab size:** 107 characters (Kannada Unicode block + punctuation + space).

---

### Section 8 — AudioAugmenter

```python
class AudioAugmenter
```

Applies data augmentation to waveforms during dataset loading.

**Augmentations:**

- **Speed perturbation** (`SPEED_PERTURB=True`): Creates 3 versions of each utterance at speeds `[0.9, 1.0, 1.1]` using `librosa.effects.time_stretch`.
- **Gaussian noise** (`NOISE_STD=0.005`): Adds small random noise to the original-speed version only, improving robustness.

---

### Section 9 — KannadaTTSDataset

```python
class KannadaTTSDataset(Dataset)
```

PyTorch Dataset that pairs text with mel spectrograms.

**Item loading:**

1. Reads text from `.txt` file
2. Loads WAV with `torchaudio.load`, resamples to `SAMPLE_RATE` if needed
3. Converts to mono, trims to `MAX_WAV_LEN`
4. Computes log-mel spectrogram: `log(mel.clamp(min=1e-5))`
5. Pads or trims mel to `MAX_MEL_LEN` frames

**Each item returns:**

| Key | Shape | Description |
| --- | --- | --- |
| `input_ids` | `(MAX_TEXT_LEN,)` | Encoded character IDs |
| `mel` | `(N_MELS, MAX_MEL_LEN)` | Log-mel spectrogram |
| `wav` | `(T_wav,)` | Raw waveform |
| `text_len` | scalar | Non-space char count + 2 (BOS+EOS), capped at `MAX_TEXT_LEN` |
| `mel_len` | scalar | Always `MAX_MEL_LEN` (mel is always padded/trimmed to this length) |

**`tts_collate`** stacks all fields into batched tensors. `wav` is kept as a list since waveform lengths vary.

---

### Section 10 — Model Architecture

#### SqueezeExcitation

```python
class SqueezeExcitation(channels, reduction=16)
```

Channel-wise attention applied after depthwise-separable convolution. Uses global average pooling followed by two 1×1 convolutions with ReLU and Sigmoid to produce per-channel scale factors.

```
Input (B, C, T)
    → AdaptiveAvgPool1d(1)     # (B, C, 1)
    → Conv1d(C, C//16, 1)      # squeeze
    → ReLU
    → Conv1d(C//16, C, 1)      # excite
    → Sigmoid
    → multiply with input      # (B, C, T)
```

---

#### DSConvBlock

```python
class DSConvBlock(channels, kernel_size, dilation=1, dropout=0.1, use_se=True)
```

Core building block used throughout encoder and decoder. Combines:

- **Depthwise convolution** — per-channel spatial filtering with optional dilation
- **Pointwise convolution** — cross-channel mixing
- **SE attention** — channel recalibration
- **Residual connection** — gradient flow
- **LayerNorm + GELU** — normalization and activation

Dilated blocks in the encoder use dilations `[1, 2, 4, 8]`, expanding the receptive field exponentially without increasing parameters.

```
Input x (B, T, C)
    → transpose → (B, C, T)
    → DepthwiseConv1d (dilation)
    → PointwiseConv1d
    → SE attention
    → transpose → (B, T, C)
    → GELU → Dropout
    → LayerNorm(x + residual)
```

---

#### PositionalEncoding

```python
class PositionalEncoding(dim, max_len=2048)
```

Sinusoidal positional encoding added to embeddings to inject sequence position information. Uses the standard sine/cosine formulation from "Attention Is All You Need". Registered as a buffer (not a trainable parameter).

---

#### DurationPredictor

```python
class DurationPredictor(embed_dim, hidden_dim=128, kernel_size=3)
```

Predicts the number of mel frames each input token should occupy.

```
Encoder output (B, T_text, C)
    → Conv1d × 2 with ReLU (hidden_dim=128)
    → transpose → (B, T_text, hidden_dim)
    → Linear → (B, T_text, 1) → squeeze → (B, T_text)
    → F.softplus(x) + 0.5     # ensures positive durations ≥ 0.5
```

`Softplus` is used instead of `ReLU` to produce smooth, always-positive duration values.

---

#### LengthRegulator

```python
class LengthRegulator
```

Expands the encoder output from text length to mel length by repeating each token's representation according to its predicted duration.

**Forward logic (per batch item):**

1. For each token at position `t`, repeat its embedding `round(duration[t])` times
2. Concatenate all repeated embeddings along the time axis
3. Pad to `mel_max_len` or truncate if over length

---

#### CNNTextEncoder

```python
class CNNTextEncoder(vocab_size, embed_dim, enc_layers, dilated_layers, kernel_size, dropout)
```

Encodes character sequences into contextual representations.

```
input_ids (B, T_text)
    → Embedding (B, T_text, EMBED_DIM)
    → PositionalEncoding
    → Standard DSConvBlock × ENC_LAYERS (dilation=1)
    → Dilated DSConvBlock × DILATED_LAYERS (dilation=1,2,4,8)
    → LayerNorm
    → (B, T_text, EMBED_DIM)
```

The combination of standard and dilated blocks gives the encoder both local phoneme-level and long-range prosodic context.

---

#### CNNMelDecoder

```python
class CNNMelDecoder(embed_dim, n_mels, dec_layers, kernel_size, dropout)
```

Refines the length-regulated encoder output into a mel spectrogram.

```
Length-regulated input (B, mel_len, EMBED_DIM)
    → DSConvBlock × DEC_LAYERS
    → LayerNorm
    → Linear(EMBED_DIM → N_MELS)
    → transpose → (B, N_MELS, mel_len)
```

---

#### PostNet

```python
class PostNet(n_mels, kernel_size=5, dropout=0.1)
```

Tacotron 2 style post-processing network. Applies 5 Conv1d layers with BatchNorm and Tanh (last layer uses Identity) to predict a residual correction to the initial mel prediction.

```
mel_before (B, N_MELS, T)
    → Conv1d+BN+Tanh+Dropout × 4
    → Conv1d+BN+Dropout × 1
    → add residual (mel_before)
    → mel_after (B, N_MELS, T)
```

The model outputs both `mel_before` (decoder output) and `mel_after` (post-net refined). Loss is computed on `mel_after`.

---

#### KannadaCNNTTS

```python
class KannadaCNNTTS(cfg)
```

Top-level model combining all components.

**`forward(input_ids, mel_max_len=None, duration_targets=None)`**

| Argument | Description |
| --- | --- |
| `input_ids` | `(B, T_text)` character IDs |
| `mel_max_len` | Target mel length for length regulator (defaults to `None`) |
| `duration_targets` | Ground-truth durations; if provided and `model.training`, used instead of predicted durations |

Returns dict: `{"mel_before", "mel_after", "durations"}`

**`synthesize(input_ids)`** — `@torch.no_grad()` inference-only forward pass; calls `model.eval()` internally, returns `mel_after` only.

---

### Section 11 — Loss Functions

#### MelLoss

```python
class MelLoss
```

Combined spectral loss with three components:

| Component | Formula | Weight |
| --- | --- | --- |
| `mel_mse` | MSE between predicted and target mel | 0.5 |
| `mel_mae` | L1 between predicted and target mel | 0.5 |
| `sc_loss` | L1 between Frobenius norms of pred and target | 0.5 |

The spectral convergence term (`sc_loss`) penalizes overall energy mismatch between predicted and target spectrograms, encouraging the model to match the global spectral shape.

**Total loss:** `0.5 × mel_mse + 0.5 × mel_mae + 0.5 × sc_loss`

---

#### GuidedAttentionLoss

```python
class GuidedAttentionLoss(sigma=0.2)
```

Regularizes the duration predictor to produce near-uniform durations early in training, encouraging diagonal alignment between text and mel frames.

For each batch item, computes the ideal per-token duration as `mel_len / text_len` and penalizes deviation from this ideal using MSE. Applied with weight `0.1` in the total loss.

---

### Section 12 — HifiGANVocoder

```python
class HifiGANVocoder(device)
```

Neural vocoder that converts mel spectrograms to waveforms.

**Primary path:** Loads `torchaudio.pipelines.HIFIGAN_VOCODER` (pre-trained HiFi-GAN). Mel is normalized to `[0, 1]` range before passing to the generator.

**Fallback path:** If HiFi-GAN fails to load, falls back to `InverseMelScale + GriffinLim` (60 iterations).

**`vocode(mel_spec)`** — accepts `(n_mels, T)` or `(B, n_mels, T)`, returns waveform tensor.

---

### Section 13 — CNNTTSTrainer

```python
class CNNTTSTrainer(cfg, model, loader, vocab)
```

Manages the full training loop.

**Optimizer:** AdamW with peak LR `1e-3`.

**LR Schedule:** Two-phase lambda scheduler:

1. **Warmup** (first `WARMUP_STEPS` steps): linear ramp from 0 to peak LR
2. **Cosine decay** (remaining steps): cosine annealing to 0

**Mixed precision:** Uses `torch.amp.GradScaler` on CUDA with `torch.amp.autocast`. Falls back to `torch.cuda.amp` for older torch versions.

**Gradient accumulation:** Gradients are accumulated over `GRAD_ACCUMULATION=4` steps before an optimizer step, giving an effective batch size of 32.

**Per-epoch logging:**

```
Epoch XX | loss: X.XXXX | mel_mse: X.XXXX | mel_mae: X.XXXX | sc_loss: X.XXXX | attn: X.XXXX | lr: X.XXXXXX
```

**Checkpoint format** (saved as `cnn_tts_v2_epoch_XX_lossX.XXXX.pt`):

```python
{
    "epoch":     int,
    "model":     state_dict,
    "optimizer": state_dict,
    "scheduler": state_dict,
    "loss":      float,
    "vocab":     char2idx dict,
}
```

---

### Section 14 — CNNTTSInference

```python
class CNNTTSInference(cfg, vocab, checkpoint_path=None)
```

Loads a trained checkpoint and synthesizes speech from Kannada text.

**`synthesize(text, output_path)`**

1. Encodes text using `KannadaVocab.encode`
2. Runs `model.synthesize(ids)` → mel spectrogram `(1, N_MELS, mel_len)`
3. Passes mel to `HifiGANVocoder.vocode` → waveform
4. Saves to `.wav` with `soundfile`
5. Returns `(output_path, audio_numpy_array)`

---

### Section 15 — Execution Flow

The script runs 6 sequential steps when executed:

```
Step 1   DatasetPreparer.run()          — download + extract dataset
Step 1.5 DatasetAnalyzer.run()          — statistical analysis + plots
Step 2   KannadaVocab.load() / build()  — vocabulary
Step 3   KannadaTTSDataset + DataLoader — dataset pipeline
Step 4   KannadaCNNTTS construction     — model + param count
Step 5   CNNTTSTrainer.train()          — 30 epochs, saves checkpoints
Step 6   CNNTTSInference.synthesize()   — 3 test sentences → .wav + playback
```

---

## Data Flow Diagram

```
Text Input (Kannada string)
        │
        ▼
KannadaVocab.encode()
  "ನಮಸ್ಕಾರ" → [1, 66, 68, 79, 93, 72, 35, 73, 2, 0, 0, ...]
        │
        ▼
CNNTextEncoder
  Embedding → PosEnc → DSConv×4 → DilatedDSConv×4
  (B, T_text, 256)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
DurationPredictor                   (encoder features)
  (B, T_text) durations
        │
        ▼
LengthRegulator
  Token expansion → (B, mel_len, 256)
        │
        ▼
CNNMelDecoder
  DSConv×4 → Linear → (B, 80, mel_len)
        │
        ▼
PostNet
  Conv1d×5 + residual → mel_after (B, 80, mel_len)
        │
        ▼
HifiGANVocoder
  mel → waveform (T_wav,)
        │
        ▼
soundfile.write() → output.wav
```

---

## Checkpoint Naming Convention

```
cnn_tts_v2_epoch_XX_lossY.YYYY.pt
```

- `XX` — zero-padded epoch number
- `Y.YYYY` — average training loss for that epoch

The best checkpoint has the **lowest loss value** in the filename. `cnn_test_analysis.py` parses this value directly to select the best checkpoint automatically.

---

## Dependencies

| Package | Purpose |
| --- | --- |
| `torch` | Model, training, tensors |
| `torchaudio` | Mel spectrogram, HiFi-GAN, resampling |
| `librosa` | Audio loading, augmentation, analysis |
| `soundfile` | WAV file writing |
| `pandas` | TSV metadata parsing |
| `numpy` | Numerical operations |
| `matplotlib` | Dataset analysis plots |
| `requests` / `tqdm` | Dataset download with progress |
| `transformers` | (Available for downstream use) |
| `pesq` / `pystoi` | Speech quality metrics (used in test analysis) |
| `jiwer` | Word error rate (available for downstream use) |

---

## Usage in Colab Notebook

This file is **Cell 1** of the Colab notebook. After it completes, the following objects are live in the kernel's memory and are directly reused by **Cell 2** (`cnn_test_analysis.py`):

| Object | Type | Used by Cell 2 |
| --- | --- | --- |
| `cfg` | `Config` | All paths, audio params, model arch |
| `vocab` | `KannadaVocab` | Text encoding |
| `dataset.items` | `list` | 100 test sample selection |
| `KannadaCNNTTS` | class | Model reconstruction |
| `CNNTTSInference` | class | Synthesis |
| `HifiGANVocoder` | class | Waveform generation |

Cell 2 installs only the additional metric packages (`pesq`, `pystoi`), adds `OUT_DIR`, defines metric functions, and runs the 100-sample evaluation loop — without re-downloading data, re-defining classes, or re-training.
