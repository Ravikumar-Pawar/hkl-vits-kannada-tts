
# HKL-VITS Hybrid Kannada TTS

A Hybrid Linguistic Kannada Text-to-Speech (TTS) system designed to generate natural Kannada speech from text using a combination of linguistic processing and VITS-based speech synthesis.

HKL-VITS integrates Kannada-specific linguistic understanding with a modern neural speech generation pipeline to improve pronunciation, speech flow, and naturalness.

---

## Quick Setup

Follow these steps to run HKL-VITS locally.

### 1. Clone Repository

```bash
git clone https://github.com/Ravikumar-Pawar/hkl-vits-kannada-tts.git

cd hkl-vits-kannada-tts
```

---

### 2. Download Model Checkpoint

Download the trained checkpoint from Google Drive:

https://drive.google.com/file/d/1-Q6tD4VsdSi3VJcRQ_DYMy2LIDsZ1BXD/view?usp=sharing

After downloading, place the checkpoint file inside:

```
checkpoints/
```

If the folder does not exist, create it manually in the project root:

```
hkl-vits-kannada-tts/
│
├── checkpoints/
│   └── model_checkpoint.pt
│
├── app/
├── docs/
├── generate_model.py
├── run.py
└── requirements.txt
```

---

### 3. Install Dependencies

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

### 4. Run Application

Start the HKL-VITS server:

```bash
python run.py
```

Example:

```powershell
(.venv) PS C:\Users\techk\Desktop\saniya\hkl-vits-hybrid-kannada-tts> python run.py
```

Server:

```
http://localhost:8080
```

Open the URL in your browser, enter Kannada text, and generate speech.





## Overview

Kannada TTS systems require handling complex linguistic properties such as:

- Kannada script representation
- Pronunciation variations
- Phoneme conversion
- Sentence rhythm
- Speech prosody

This project introduces a hybrid approach by combining:

- Kannada text normalization
- Grapheme-based linguistic encoding
- Phoneme-based linguistic encoding
- Feature fusion mechanism
- Prosody prediction
- MMS Kannada VITS speech synthesis backbone

The implementation provides an end-to-end pipeline from Kannada text input to generated speech output.

---

## Implementation Highlights

The system is designed with a modular deep learning architecture:

- Kannada-aware preprocessing pipeline
- Custom linguistic encoder architecture
- Grapheme and phoneme feature processing
- Feature fusion mechanism
- Prosody modeling for:
  - Pause prediction
  - Duration prediction
  - Pitch prediction
  - Energy prediction
- FastAPI inference backend
- Training and checkpoint generation pipeline

The implementation focuses on improving Kannada speech generation by adding language-specific features on top of an existing neural TTS backbone.

---

## Features

- Kannada text-to-speech generation
- Neural speech synthesis using VITS architecture
- Kannada linguistic processing
- Custom HKL hybrid encoder design
- Dataset preparation pipeline
- Model training pipeline
- Checkpoint generation
- FastAPI-based API service
- Web interface for testing

---

## Running Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python run.py
```

Server:

```
http://localhost:8080
```

Enter Kannada text in the web interface and generate speech.

---

## Model Generation

To prepare the dataset and start model generation:

```bash
python generate_model.py
```

The pipeline performs:

```
Dataset Preparation
        |
        ↓
Text Processing
        |
        ↓
Training Pipeline
        |
        ↓
Checkpoint Generation
```

Generated checkpoints are stored inside:

```
checkpoints/
```

---

## Project Documentation

Complete technical documentation, architecture explanation, implementation details, and research report are available in:

```
docs/report/
```

Documentation includes:

### `report.md`

Complete project report.

### `part1.md`

System design and architecture.

### `part2.md`

Model components and implementation details.

### `part3.md`

Training pipeline and technical explanation.

### `Comparison.md`

Comparison with existing approaches.

For detailed understanding of the architecture and implementation, refer to the documentation folder.

---

## Technologies Used

- Python
- PyTorch
- FastAPI
- Transformers
- HuggingFace Models
- MMS VITS
- Neural Speech Processing

---

## Future Improvements

Possible improvements:

- Better Kannada phoneme alignment
- Multi-speaker Kannada TTS
- Emotion-controlled speech generation
- Larger Kannada speech datasets
- Improved end-to-end fine-tuning

---

## License

This project is developed for research and educational purposes.
