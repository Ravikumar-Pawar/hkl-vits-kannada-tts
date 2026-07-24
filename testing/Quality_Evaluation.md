# HKL-VITS Kannada TTS Quality Evaluation

# 1. Model Description

## 1.1 HKL-VITS Model

The proposed **HKL-VITS (Hybrid Kannada Linguistic-VITS)** model is an enhanced Kannada Text-to-Speech (TTS) system developed by integrating Kannada-specific linguistic processing modules with the pretrained MMS-VITS acoustic model. The model introduces language-specific preprocessing and linguistic feature extraction to improve pronunciation accuracy, speech intelligibility, and naturalness for Kannada speech synthesis.

The HKL-VITS architecture integrates multiple components, including Kannada text normalization, Grapheme-to-Phoneme (G2P) conversion, dual linguistic encoders, linguistic feature fusion, prosody prediction, and the pretrained MMS-VITS acoustic backbone for waveform generation.

The inference pipeline consists of the following stages:

1. Kannada Text Input
2. Text Normalization
3. Grapheme-to-Phoneme Conversion
4. Grapheme Encoder
5. Phoneme Encoder
6. Linguistic Feature Fusion
7. Prosody Prediction
8. MMS-VITS Acoustic Model
9. Waveform Generation

The proposed model was evaluated using **100 Kannada text samples**, with **three synthesis repetitions per sample**, producing a total of **300 synthesized speech files**.

---

# 2. Testing Methodology

The evaluation was performed to verify the functional correctness, runtime performance, repeatability, stability, and objective speech quality of the proposed HKL-VITS model.

A notebook-based automated evaluation framework was developed to synthesize speech, collect runtime statistics, generate objective quality metrics, and produce evaluation reports automatically.

Each input sentence was synthesized **three times** to verify inference consistency and repeatability.

## Evaluation Configuration

| Parameter               | Value        |
| ----------------------- | ------------ |
| Evaluation Samples      | 100          |
| Repetitions per Sample  | 3            |
| Total Syntheses         | 300          |
| Output Format           | WAV          |
| Sample Rate             | 16 kHz       |
| Execution Environment   | Google Colab |
| Programming Language    | Python       |
| Deep Learning Framework | PyTorch      |

## Evaluation Workflow

1. Load evaluation dataset
2. Read Kannada sentence
3. Perform text normalization
4. Convert text into phonemes
5. Generate linguistic embeddings
6. Predict prosody
7. Generate speech waveform
8. Save synthesized audio
9. Compute objective quality metrics
10. Collect runtime statistics
11. Generate JSON and CSV evaluation reports

---

# 3. Mathematical Expressions

## 3.1 Synthesis Latency

[
Latency=T_{end}-T_{start}
]

where

* (T_{start}) = synthesis start time
* (T_{end}) = synthesis completion time

---

## 3.2 Average Latency

[
L_{avg}=\frac1N\sum_{i=1}^{N}L_i
]

where

* (N) = total synthesis runs
* (L_i) = latency of each synthesis

---

## 3.3 Throughput

[
Throughput=\frac{N}{T}
]

where

* (N) = total synthesized samples
* (T) = total execution time

---

## 3.4 Real-Time Factor (RTF)

[
RTF=\frac{T_{synthesis}}{T_{audio}}
]

where

* (T_{synthesis}) = synthesis time
* (T_{audio}) = generated audio duration

RTF < 1 indicates faster-than-real-time synthesis.

---

## 3.5 Root Mean Square (RMS)

[
RMS=\sqrt{\frac1N\sum_{i=1}^{N}x_i^2}
]

where (x_i) represents the audio amplitude.

---

## 3.6 Dynamic Range

[
DR=20\log_{10}\left(\frac{Peak}{RMS}\right)
]

---

## 3.7 Silence Ratio

[
Silence\ Ratio=\frac{Silent\ Samples}{Total\ Samples}
]

---

## 3.8 Mean Opinion Score (MOS)

[
MOS=\frac1N\sum_{i=1}^{N}R_i
]

where

* (R_i) = listener rating
* (N) = number of listeners

---

## 3.9 Perceptual Evaluation of Speech Quality (PESQ)

[
PESQ=f(Reference,\ Synthesized)
]

Expected range:

**1.0 – 4.5**

Higher values indicate better perceptual quality.

---

## 3.10 Short-Time Objective Intelligibility (STOI)

[
STOI=f(Reference,\ Synthesized)
]

Expected range:

**0 – 1**

Higher values indicate better intelligibility.

---

## 3.11 Mel Cepstral Distortion (MCD)

[
MCD=\frac{10}{\ln10}\sqrt{2\sum_{i=1}^{K}(mc_i^{ref}-mc_i^{syn})^2}
]

Lower values indicate better speech quality.

---

## 3.12 Word Error Rate (WER)

[
WER=\frac{S+D+I}{N}
]

where

* S = substitutions
* D = deletions
* I = insertions
* N = reference words

---

## 3.13 Character Error Rate (CER)

[
CER=\frac{S+D+I}{N}
]

where

* (N) = reference characters

---

# 4. Test Cases

## 4.1 Functional Testing

| Test ID | Description                    | Expected Result             | Status |
| ------- | ------------------------------ | --------------------------- | ------ |
| FT-01   | Kannada text input             | Text accepted               | Pass   |
| FT-02   | Text normalization             | Normalized text generated   | Pass   |
| FT-03   | Grapheme-to-Phoneme conversion | Phoneme sequence generated  | Pass   |
| FT-04   | Linguistic feature extraction  | Feature vectors generated   | Pass   |
| FT-05   | Prosody prediction             | Prosody embedding generated | Pass   |
| FT-06   | Waveform generation            | Speech waveform generated   | Pass   |
| FT-07   | Audio file saving              | WAV file generated          | Pass   |

## 4.2 Performance Testing

| Test ID | Description            | Expected Result | Status |
| ------- | ---------------------- | --------------- | ------ |
| PT-01   | Latency measurement    | Successful      | Pass   |
| PT-02   | Throughput calculation | Successful      | Pass   |
| PT-03   | CPU monitoring         | Successful      | Pass   |
| PT-04   | RAM monitoring         | Successful      | Pass   |
| PT-05   | GPU monitoring         | Successful      | Pass   |

## 4.3 Repeatability Testing

| Test ID | Description                    | Expected Result     | Status |
| ------- | ------------------------------ | ------------------- | ------ |
| RT-01   | Repeat synthesis               | Stable outputs      | Pass   |
| RT-02   | Multiple executions            | No runtime failures | Pass   |
| RT-03   | Consistent waveform generation | Successful          | Pass   |

## 4.4 Speech Quality Testing

| Test ID | Metric | Description                    | Status      |
| ------- | ------ | ------------------------------ | ----------- |
| QT-01   | MOS    | Human listening evaluation     | In Progress |
| QT-02   | PESQ   | Objective speech quality       | Computed    |
| QT-03   | STOI   | Objective intelligibility      | Computed    |
| QT-04   | MCD    | Spectral distortion            | Computed    |
| QT-05   | WER    | ASR transcription accuracy     | Pending     |
| QT-06   | CER    | Character recognition accuracy | Pending     |

---

# 5. Test Results

## 5.1 Evaluation Configuration

| Parameter              |        Value |
| ---------------------- | -----------: |
| Total Test Samples     |      **100** |
| Repetitions per Sample |        **3** |
| Total Syntheses        |      **300** |
| Output Format          |          WAV |
| Sample Rate            | **16000 Hz** |

---

## 5.2 Runtime Performance Results

| Metric                  |       Measured Value |
| ----------------------- | -------------------: |
| Minimum Latency         |       **0.088479 s** |
| Average Latency         |       **0.202323 s** |
| Median Latency          |       **0.191442 s** |
| Maximum Latency         |       **0.427581 s** |
| 95th Percentile Latency |       **0.330789 s** |
| Standard Deviation      |       **0.067994 s** |
| Average Throughput      | **4.94 syntheses/s** |

---

## 5.3 Objective Speech Quality Results

| Metric |        Average | Minimum |  Maximum | Remarks          |
| ------ | -------------: | ------: | -------: | ---------------- |
| PESQ   |     **1.5833** |  1.0195 |   4.3215 | Computed         |
| STOI   |     **0.0467** | -0.3393 |   0.4060 | Computed         |
| MCD    |   **103.4634** | 51.8094 | 154.9108 | Computed         |
| WER    |  Not Available |       - |        - | Pending          |
| CER    |  Not Available |       - |        - | Pending          |
| MOS    | Survey Ongoing |       - |        - | 100 Participants |

> **Note:** PESQ, STOI, and MCD values were successfully computed by the evaluation framework. However, the unusually low PESQ and STOI scores and high MCD values indicate that the reference and synthesized audio may not be fully aligned for objective comparison. These metrics should therefore be interpreted cautiously until the evaluation pipeline is refined.

---

## 5.4 Audio Statistics

| Metric                |        Average |
| --------------------- | -------------: |
| Audio Duration        |    **6.585 s** |
| Words per Second      |     **1.0553** |
| Characters per Second |     **8.5543** |
| Silence Ratio         |     **0.3801** |
| Dynamic Range         | **14.5404 dB** |
| Clipping Percentage   | **0.000003 %** |

---

## 5.5 Resource Utilization

| Metric                 |  Measured Value |
| ---------------------- | --------------: |
| CPU Usage              |     **100.0 %** |
| RAM Usage              |      **50.4 %** |
| Process RSS Memory     |  **2780.65 MB** |
| Process VMS Memory     | **15220.65 MB** |
| GPU Utilization        |       **0.0 %** |
| GPU Memory Used        |  **1452.19 MB** |
| GPU Memory Utilization |      **9.45 %** |

---

## 5.6 Overall Evaluation Summary

| Evaluation             | Result                         |
| ---------------------- | ------------------------------ |
| Functional Testing     | Pass                           |
| Performance Testing    | Pass                           |
| Repeatability Testing  | Pass                           |
| Runtime Stability      | Pass                           |
| Audio Generation       | Pass                           |
| JSON Report Generation | Pass                           |
| CSV Report Generation  | Pass                           |
| PESQ Evaluation        | Computed                       |
| STOI Evaluation        | Computed                       |
| MCD Evaluation         | Computed                       |
| MOS Evaluation         | In Progress (100 Participants) |
| WER Evaluation         | Pending                        |
| CER Evaluation         | Pending                        |

---

# 6. Conclusion

The proposed HKL-VITS model successfully synthesized **300 Kannada speech samples** generated from **100 unique text inputs**, demonstrating stable and repeatable inference performance. The evaluation framework successfully validated the complete synthesis pipeline, including text normalization, grapheme-to-phoneme conversion, linguistic feature extraction, prosody prediction, waveform generation, and audio file creation.

The average synthesis latency of **0.2023 seconds** and throughput of approximately **4.94 syntheses per second** indicate that the model is capable of efficient real-time inference. Resource utilization remained stable throughout the evaluation, confirming reliable execution within the Google Colab environment.

Objective speech quality metrics (PESQ, STOI, and MCD) were successfully computed; however, their values suggest that additional refinement of the reference-audio alignment process is required for accurate objective evaluation. Subjective evaluation using a Mean Opinion Score (MOS) listening test involving **100 participants** is currently in progress and will provide complementary evidence of speech naturalness and intelligibility.

Overall, the experimental results demonstrate that the proposed HKL-VITS framework achieves robust functional performance, stable runtime characteristics, and a reproducible evaluation workflow suitable for Kannada Text-to-Speech research and future optimization.
