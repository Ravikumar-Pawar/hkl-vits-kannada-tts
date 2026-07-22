
# HKL-VITS Overall System Test Cases

| Test ID | Test Scenario              | Input                        | Expected Result                             | Status |
| ------- | -------------------------- | ---------------------------- | ------------------------------------------- | ------ |
| TC-001  | Kannada sentence synthesis | `ನಮಸ್ಕಾರ`                    | Speech generated successfully               | Pass   |
| TC-002  | Long Kannada paragraph     | 200+ character paragraph     | Complete speech generated without errors    | Pass   |
| TC-003  | Number normalization       | `2500`                       | Number expanded and spoken correctly        | Pass   |
| TC-004  | Currency normalization     | `₹2500`                      | Spoken as "ರೂಪಾಯಿ ಎರಡು ಸಾವಿರದ ಐನೂರು"        | Pass   |
| TC-005  | Percentage normalization   | `50%`                        | Spoken as "ಐವತ್ತು ಶೇಕಡಾ"                    | Pass   |
| TC-006  | Symbol normalization       | `A+B=C`                      | Symbols expanded correctly                  | Pass   |
| TC-007  | Mixed Kannada-English text | `ಬೆಂಗಳೂರು City`              | Speech generated without failure            | Pass   |
| TC-008  | Conjunct consonants        | `ಕ್ಷೇತ್ರ`                    | Correct pronunciation of "ಕ್ಷ"              | Pass   |
| TC-009  | Geminated consonants       | `ಅಕ್ಕ`                       | Correct pronunciation of doubled consonants | Pass   |
| TC-010  | Long vowel pronunciation   | `ಕೀ`, `ಕೂ`                   | Correct vowel duration                      | Pass   |
| TC-011  | Punctuation handling       | `ನೀವು ಹೇಗಿದ್ದೀರಿ?`           | Natural pause and intonation                | Pass   |
| TC-012  | Empty input                | `""`                         | Validation message, no crash                | Pass   |
| TC-013  | Special characters only    | `@#$%^`                      | Graceful handling without exception         | Pass   |
| TC-014  | Unicode Kannada text       | Kannada Unicode sentence     | Correct synthesis                           | Pass   |
| TC-015  | Unsupported characters     | Emoji or unsupported symbols | Ignored or handled safely                   | Pass   |

---

# End-to-End Pipeline Validation

| Test ID | Test Description                                  | Expected Result                   |
| ------- | ------------------------------------------------- | --------------------------------- |
| TC-016  | Complete preprocessing → speech pipeline          | Pipeline completes successfully   |
| TC-017  | Text normalization → G2P → HKL encoder → MMS-VITS | All stages execute sequentially   |
| TC-018  | Waveform generation                               | Valid waveform produced           |
| TC-019  | Audio file creation                               | WAV file saved successfully       |
| TC-020  | Audio playback                                    | Audio playable without corruption |

---

# Model Validation

| Test ID | Description                  | Expected Result                      |
| ------- | ---------------------------- | ------------------------------------ |
| TC-021  | Grapheme and phoneme fusion  | Features generated successfully      |
| TC-022  | Prosody prediction           | Pitch, duration and energy predicted |
| TC-023  | Emotion synthesis (Happy)    | Emotion reflected in speech          |
| TC-024  | Emotion synthesis (Sad)      | Emotion reflected in speech          |
| TC-025  | Emotion synthesis (Question) | Rising intonation produced           |

---

# Dataset Validation

| Test ID | Description        | Expected Result                |
| ------- | ------------------ | ------------------------------ |
| TC-026  | Dataset loading    | Dataset loads successfully     |
| TC-027  | Audio-text pairing | All WAV files match text files |
| TC-028  | Missing audio file | Error handled gracefully       |
| TC-029  | Missing transcript | Sample skipped or reported     |
| TC-030  | Invalid audio      | Corrupt file detected          |

---

# Performance Testing

| Test ID | Description               | Acceptance Criteria                      |
| ------- | ------------------------- | ---------------------------------------- |
| TC-031  | Single sentence inference | < 5 seconds                              |
| TC-032  | Large paragraph inference | Completes without memory issues          |
| TC-033  | GPU utilization           | GPU used when available                  |
| TC-034  | CPU execution             | Runs successfully on CPU                 |
| TC-035  | Memory usage              | No memory leak during repeated inference |

---

# Robustness Testing

| Test ID | Description                                          | Expected Result           |
| ------- | ---------------------------------------------------- | ------------------------- |
| TC-036  | Very long sentence (1000+ chars)                     | No crash                  |
| TC-037  | Multiple consecutive requests                        | Stable execution          |
| TC-038  | Repeated synthesis                                   | Consistent output         |
| TC-039  | Network unavailable during training dataset download | Appropriate error message |
| TC-040  | Invalid configuration                                | Validation error returned |

---

# Speech Quality Evaluation

| Test ID | Evaluation             | Method                   |
| ------- | ---------------------- | ------------------------ |
| TC-041  | Pronunciation accuracy | Human evaluation         |
| TC-042  | Naturalness            | Mean Opinion Score (MOS) |
| TC-043  | Intelligibility        | Listener evaluation      |
| TC-044  | Prosody quality        | Expert evaluation        |
| TC-045  | Emotion correctness    | Listener evaluation      |

---

# Compatibility Testing

| Test ID | Description            | Expected Result             |
| ------- | ---------------------- | --------------------------- |
| TC-046  | Google Colab execution | Successful                  |
| TC-047  | Linux execution        | Successful                  |
| TC-048  | CUDA GPU execution     | Successful                  |
| TC-049  | CPU-only execution     | Successful                  |
| TC-050  | PyTorch compatibility  | Runs with supported version |

---

# Overall Acceptance Criteria

| Requirement                       | Expected Outcome |
| --------------------------------- | ---------------- |
| Kannada text accepted             | Yes              |
| Numbers normalized                | Yes              |
| Symbols expanded                  | Yes              |
| G2P conversion successful         | Yes              |
| Linguistic features generated     | Yes              |
| Prosody predicted                 | Yes              |
| Emotion conditioning applied      | Yes (if enabled) |
| MMS-VITS synthesis successful     | Yes              |
| WAV file generated                | Yes              |
| Audio is intelligible and natural | Yes              |

### Testing Tools

| Testing Type              | Tool                                            |
| ------------------------- | ----------------------------------------------- |
| Functional/System Testing | `pytest`                                        |
| API Testing (FastApi)       | Postman, pytest                                 |
| Performance Testing       | `torch.profiler`, `cProfile`, `memory_profiler` |
| Load Testing              | Locust or Apache JMeter                         |
| Audio Quality Evaluation  | MOS, PESQ, STOI, Mel-Cepstral Distortion (MCD)  |
| Experiment Tracking       | TensorBoard, MLflow or Weights & Biases         |

