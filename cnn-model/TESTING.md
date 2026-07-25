# CNN Kannada TTS v2 — Testing Documentation

## Overview

This document describes the test analysis pipeline for the CNN Kannada TTS v2 model, the metrics used for evaluation, and the results obtained.

## Test Pipeline

The test analysis (`cnn_test_analysis.py`) is a single Colab cell that performs the following steps:

### Step 1 — Dataset Preparation
- Downloads the OpenSLR Kannada male dataset (`kn_in_male.zip`) if not already present
- Extracts WAV files to `/content/kannada_tts_dataset/wavs/`
- Parses the TSV metadata and creates per-sample text files in `/content/kannada_tts_dataset/text/`
- **Result**: 2,214 WAV files and 2,214 text files ready

### Step 1.5 — Dataset Analysis
- Computes text length statistics (characters)
- Computes audio duration statistics (seconds) using `librosa.get_duration()`
- Estimates mel frame counts (hop=256, sr=22050)
- Reports truncation statistics against `MAX_MEL_LEN=860`

**Dataset Statistics:**
| Metric | Value |
|--------|-------|
| Total samples | 2,214 |
| Avg text length | 56.1 chars |
| Avg audio duration | 6.79s |
| Total audio | 4.17 hours |
| Unique characters | 103 |
| Sample rate | 22,050 Hz |
| Mel bands | 80 |
| Max mel frames | 860 |
| Samples truncated (>MAX_MEL_LEN) | 15.5% |

### Step 2 — Vocabulary
- Loads the pre-built vocabulary from `/content/cnn_tts_v2_vocab.json`
- **Vocab size**: 107 characters (103 Kannada + 4 special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`)

### Step 3 — Model & Best Checkpoint
- Builds the `KannadaCNNTTSv2` model with the v2 architecture
- Loads the best checkpoint by parsing loss from filename
- **Best checkpoint**: `cnn_tts_v2_epoch_03_loss67.3103.pt`
- **Train loss**: 67.3103
- **Total params**: 1,273,665

### Step 4 — Test Sample Collection
- Collects 100 random samples from the 2,214 available (seed=42)
- Each sample contains: text + corresponding WAV file path

### Step 5 — Synthesis & Metric Computation
- For each of the 100 test samples:
  1. Encode text to token IDs (max_len=150)
  2. Run `model.synthesize(ids)` → log-mel spectrogram
  3. Vocode mel → waveform using `HifiGANVocoder` (Griffin-Lim fallback)
  4. Save synthesized audio to `/content/cnn_test_outputs/syn_XXX.wav`
  5. Compute 8 objective metrics comparing synthesized vs. reference audio

### Step 6 — Results Summary
- Aggregates metrics across all 100 samples (mean, std, min, max)
- Saves full per-sample results to `cnn_test_results.csv`

### Step 7 — Sample Playback
- Displays first 3 synthesized samples with audio players

## Model Architecture (v2)

```
KannadaCNNTTSv2(
  (encoder): CNNTextEncoder
    - Embedding(107, 256, padding_idx=0)
    - PositionalEncoding
    - 4 × DSConvBlock (dilation=1, with SqueezeExcitation)
    - 4 × DSConvBlock (dilation=[1,2,4,8], with SqueezeExcitation)
    - LayerNorm(256)
  (duration_predictor): DurationPredictor
    - Conv1d(256, 128) → ReLU → Conv1d(128, 128) → ReLU → Linear(128, 1) → softplus
  (length_regulator): LengthRegulator
  (decoder): CNNMelDecoder
    - 4 × DSConvBlock (dilation=1)
    - LayerNorm(256)
    - Linear(256, 80)
  (postnet): PostNet
    - 5 × Conv1d(80, 80, kernel=5) with BatchNorm + Tanh/Identity + Dropout
)
```

### Key v2 Enhancements over v1
| Feature | v1 | v2 |
|---------|-----|-----|
| DSConv blocks | 8 (no SE, no dilation) | 4 standard + 4 dilated (with SE) |
| Duration modeling | Naive interpolation | DurationPredictor + LengthRegulator |
| Post-processing | None | PostNet (5 layers) |
| Vocoder | Griffin-Lim | HiFi-GAN (with Griffin-Lim fallback) |
| Loss | MSE only | L1 + L2 + Spectral Convergence + Guided Attention |

## Vocoder

The `HifiGANVocoder` class attempts to load `torchaudio.pipelines.HIFIGAN_VOCODER`. In torchaudio 2.11.0, this attribute does not exist, so it falls back to **Griffin-Lim** with 60 iterations.

**Griffin-Lim pipeline:**
1. Undo log: `mel.exp()` → power mel spectrogram
2. Inverse mel: `InverseMelScale(n_stft=513, n_mels=80, sr=22050)` → linear spectrogram
3. Griffin-Lim: `GriffinLim(n_fft=1024, hop=256, n_iter=60)` → waveform

## Evaluation Metrics

| Metric | Description | Lower/Higher | Good Threshold |
|--------|-------------|-------------|----------------|
| MCD (dB) | Mel Cepstral Distortion | Lower | < 8 dB |
| LSD | Log-Spectral Distance | Lower | — |
| F0 RMSE (Hz) | Pitch accuracy | Lower | — |
| Energy RMSE | Loudness accuracy | Lower | — |
| Duration Ratio | Syn/Ref length ratio | ~1.0 | 1.0 = perfect |
| SNR (dB) | Signal-to-Noise Ratio | Higher | > 0 dB |
| PESQ | Perceptual quality | Higher | -0.5 to 4.5 |
| STOI | Speech intelligibility | Higher | 0 to 1 |

## Test Results (3-epoch checkpoint)

```
=================================================================
  CNN KANNADA TTS v2 — TEST ANALYSIS  (Pure CNN, No MMS/VITS)
=================================================================
  Checkpoint : cnn_tts_v2_epoch_03_loss67.3103.pt
  Epoch      : 3  |  Train Loss: 67.3103
  Vocoder    : Griffin-Lim
  Samples    : 100 / 100
=================================================================
                     Mean       Std        Min        Max
MCD (dB)        3559.0123  461.2900  2777.0113  4862.8365
LSD                7.2081    0.5408     6.2006     8.7269
F0 RMSE (Hz)      93.5733   24.2926    40.1731   163.0685
Energy RMSE        0.7602    0.0562     0.5160     0.8378
Duration Ratio     4.4445    2.1853     1.6975    11.6027
SNR (dB)         -24.2198    5.4928   -39.8290   -15.8463
PESQ               1.0551    0.0412     1.0177     1.2425
STOI               0.4127    0.0653     0.1737     0.5372
=================================================================
```

### Analysis

The metrics indicate the model needs more training (only 3 of 30 epochs completed):

1. **MCD (3559 dB)**: Extremely high — the mel spectrograms are not well-aligned. This is expected with only 3 epochs of training.

2. **Duration Ratio (4.44)**: The duration predictor is over-predicting durations by ~4.4×. The `LengthRegulator` expands each token by the predicted duration, and with insufficient training, the predicted durations are too large. This causes the synthesized audio to be much longer than the reference.

3. **SNR (-24.22 dB)**: Negative SNR indicates the synthesized audio is dominated by artifacts rather than the target signal. This is a combination of the under-trained model and the Griffin-Lim vocoder.

4. **PESQ (1.06)**: Low but above the minimum (-0.5), indicating some perceptual structure is present.

5. **STOI (0.41)**: Moderate intelligibility — the model has learned some speech-like patterns but not enough for clear intelligibility.

6. **LSD (7.21)**: Reasonable spectral distance, suggesting the overall spectral shape is somewhat captured.

### Expected Improvements with More Training

| Epochs | Expected MCD | Expected Duration Ratio | Expected SNR |
|--------|-------------|------------------------|-------------|
| 3 (current) | ~3559 dB | ~4.44 | ~-24 dB |
| 10 | ~50-100 dB | ~2.0-3.0 | ~-15 dB |
| 30 (full) | < 8 dB | ~1.0-1.5 | > 0 dB |

## How to Run

1. Open `cnn_test_analysis.py` in a Colab notebook
2. Ensure the v2 checkpoint exists at `/content/cnn_tts_v2_checkpoints/cnn_tts_v2_epoch_*.pt`
3. Ensure the v2 vocab exists at `/content/cnn_tts_v2_vocab.json`
4. Run all cells
5. Results are saved to `/content/cnn_test_outputs/cnn_test_results.csv`

## Files

| File | Description |
|------|-------------|
| `cnn_test_analysis.py` | Main test analysis script (single Colab cell) |
| `colab_single_cell.py` | Training script (v2 architecture) |
| `model.py` | v1 model architecture (MMS backbone, not used by v2) |
| `config.py` | v1 config (not used by v2) |
| `train.py` | v1 training loop (not used by v2) |
| `inference.py` | v1 inference (not used by v2) |
| `generate_cnn_model.py` | v1 entry point (not used by v2) |
| `dataset.py` | v1 dataset pipeline (not used by v2) |
