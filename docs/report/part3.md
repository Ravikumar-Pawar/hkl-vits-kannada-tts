# HKL-VITS Implementation Details Report

# Part 3 — Fine-Tuned MMS-VITS Integration, Training Pipeline, Parameters, Inference, and Research Comparison

---

# 12. Module 8 — Fine-Tuned MMS-VITS Acoustic Backbone

## Purpose

The final stage of HKL-VITS converts linguistic information into actual Kannada speech waveform.

The previous modules only understand language:

```
Text
 |
G2P
 |
Grapheme Encoder
 |
Phoneme Encoder
 |
Fusion
 |
Prosody
```

Output:

```
Linguistic Features
```

But this is not audio yet.

A neural speech synthesizer is required.

For this purpose HKL-VITS integrates a fine-tuned:

```
MMS-VITS Kannada Acoustic Model
```

---

# Why MMS-VITS was selected?

Training a complete TTS model from zero requires:

* thousands of hours of speech
* powerful GPUs
* long training time
* complex alignment learning

MMS-VITS already provides:

* VITS architecture
* neural acoustic generation
* waveform synthesis
* Kannada language capability

Instead of rebuilding VITS, HKL-VITS improves the linguistic front-end.

---

## Original VITS Pipeline

Traditional VITS:

```
Text
 |
Text Encoder
 |
Posterior Encoder
 |
Flow
 |
HiFi-GAN Decoder
 |
Audio
```

The text encoder is generic.

For Kannada this creates limitations.

---

## HKL-VITS Modified Pipeline

```
Kannada Text

     |
     |

HKL Linguistic Front-End

     |
     |

Hybrid Representation

     |
     |

Adapter Layer

     |
     |

Fine-tuned MMS-VITS

     |
     |

Waveform

```

---

# 13. MMS Adapter Layer

The HKL encoder dimension and MMS dimension are different.

Example:

HKL:

```
hidden size = 256
```

MMS:

```
hidden size = 192
```

They cannot directly connect.

Therefore:

```
HKL Feature

[batch, tokens, 256]


        |
        |

Linear Projection


        |
        |


MMS Feature

[batch,tokens,192]

```

Implementation concept:

```python
nn.Linear(256,192)
```

Purpose:

* feature compatibility
* dimensional conversion
* preserve learned linguistic information

---

# 14. Complete HKL-VITS Data Flow

Final complete architecture:

```
                 Input Kannada Text

                         |

             Text Normalization

                         |

              Kannada G2P Engine

                         |

          ----------------------------

          |                          |

   Grapheme Encoder           Phoneme Encoder

    Transformer                 BiLSTM


          |                          |

          ----------------------------

                    Fusion Attention

                         |

              HKL Linguistic Feature

                    [B,T,256]

                         |

              Prosody Predictor

                         |

      ---------------------------------

      |              |                |

    Pitch        Duration          Energy

                         |

                  Emotion Layer

                         |

                Adapter Projection

                    [B,T,192]

                         |

              Fine-tuned MMS-VITS

                         |

                 Neural Decoder

                         |

                    Waveform

                         |

                 Final Kannada Audio

```

---

# 15. Training Pipeline

## Dataset

HKL-VITS uses:

Kannada Speech Dataset

Structure:

```
dataset/

 wavs/

    audio001.wav

    audio002.wav


 text/

    audio001.txt

    audio002.txt

```

Each sample:

```
Text
+
Corresponding speech
```

Example:

Text:

```
ನಾನು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇನೆ
```

Audio:

```
speaker recording
```

---

# Dataset Processing

## Step 1 — Audio Loading

Audio:

```
16 kHz sampling rate
```

Converted:

```
Waveform Tensor
```

Example:

```
[16000 samples/sec]
```

---

## Step 2 — Text Processing

Original:

```
ನಾನು 123 ಮಾತನಾಡುತ್ತೇನೆ
```

After normalization:

```
ನಾನು ಒಂದು ಎರಡು ಮೂರು ಮಾತನಾಡುತ್ತೇನೆ
```

---

## Step 3 — G2P Conversion

Output:

```
naa nu

ka nː na ɖa

maa ta naa ɖu

```

---

## Step 4 — Feature Generation

Generate:

```
Grapheme embeddings

Phoneme embeddings

Prosody targets

Emotion embeddings
```

---

# 16. HKL-VITS Training Objective

Training does not optimize only audio quality.

It learns multiple objectives.

---

## 1. Acoustic Reconstruction Loss

Purpose:

Make generated audio close to real audio.

Formula:

```
L_reconstruction

=

Generated Mel

-

Real Mel

```

Improves:

* speech clarity
* pronunciation
* voice quality

---

# 2. Prosody Loss

The model learns:

* pitch
* duration
* energy

Example:

Real:

```
ಕನ್ನಡ
```

Duration:

```
ka nː na ɖa

```

Generated:

```
ka na na da

```

Loss teaches:

```
nː must be longer
```

---

# 3. Phoneme Clarity Loss

Kannada-specific improvement.

Purpose:

Separate similar sounds.

Examples:

```
ಕಿ

vs

ಕೀ
```

Short and long vowel.

And:

```
ಕ

vs

ಕ್ಕ
```

Single vs gemination.

The model learns:

```
Different phonemes
=
Different acoustic representation
```

---

# 17. Training Configuration

Example HKL-VITS configuration:

```json
{
"model":

{

"grapheme_hidden":256,

"phoneme_hidden":256,

"mms_hidden":192,

"phoneme_encoder":"BiLSTM",

"fusion":"Multi Head Attention"

},


"training":

{

"batch_size":2,

"gradient_accumulation":8,

"learning_rate":0.0002,

"epochs":10

}

}
```

---

# 18. Why BiLSTM in HKL-VITS Prosody Model?

Speech timing depends on sequence.

Example:

Sentence:

```
ನಾನು ಇಂದು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ
```

The pause after:

```
ಇಂದು
```

depends on upcoming words.

BiLSTM sees:

Previous:

```
ನಾನು ಇಂದು
```

Future:

```
ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ
```

Therefore it predicts:

```
Pause = low
Duration = normal
Pitch = rising
```

Better than independent token prediction.

---

# 19. Inference Pipeline

User input:

```
ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ
```

---

## Step 1

Normalize:

```
same text
```

---

## Step 2

G2P:

```
naa nu

i ndu

tu mba

sa nto sha

```

---

## Step 3

Generate HKL representation:

Output:

```
[1,63,256]
```

---

## Step 4

Generate prosody:

Output:

```
pause:

[1,63]


duration:

[1,63]


pitch:

[1,63]


energy:

[1,63]

```

---

## Step 5

Emotion selection:

Example:

```
happy
```

Emotion embedding added.

---

## Step 6

MMS synthesis:

Output:

```
waveform

[70000 samples]

```

Saved:

```
hkl_happy.wav
```

---

# 20. Hardware Requirement

Development:

```
Google Colab T4

GPU:
Tesla T4

VRAM:
16GB

CUDA:
12.x

PyTorch:
2.x

```

---

Training:

Recommended:

```
Batch:
2

Gradient accumulation:
8

Effective batch:
16

Epoch:
10+

```

---

# 21. Comparison With Existing TTS Models

| Feature               | Tacotron2 | FastSpeech2 | Generic VITS | MMS-TTS | HKL-VITS    |
| --------------------- | --------- | ----------- | ------------ | ------- | ----------- |
| End-to-end            | Yes       | Partial     | Yes          | Yes     | Yes         |
| Kannada optimized     | No        | No          | No           | Basic   | Yes         |
| G2P support           | Limited   | Limited     | Basic        | Basic   | Advanced    |
| Morphology handling   | Low       | Medium      | Low          | Medium  | High        |
| Vowel length handling | Weak      | Medium      | Medium       | Medium  | Explicit    |
| Gemination handling   | Weak      | Medium      | Medium       | Medium  | Explicit    |
| Grapheme encoder      | Yes       | Yes         | Yes          | Yes     | Transformer |
| Phoneme encoder       | No        | Optional    | No           | No      | BiLSTM      |
| Dual representation   | No        | No          | No           | No      | Yes         |
| Prosody control       | Limited   | Good        | Medium       | Medium  | Advanced    |
| Emotion control       | Limited   | Optional    | Limited      | Limited | Integrated  |
| Training cost         | High      | Medium      | High         | Lower   | Lower       |
| Kannada adaptability  | Low       | Medium      | Medium       | Medium  | High        |

---

# 22. Why HKL-VITS is Different

Traditional TTS:

```
Text
 |
One encoder
 |
Audio
```

HKL-VITS:

```
Text

+

Kannada Linguistic Rules

+

G2P

+

Grapheme Understanding

+

Phoneme Understanding

+

Prosody Learning

+

Emotion Control

+

Fine-tuned VITS Generation

=

Kannada Optimized Speech
```

---

# 23. Research Contribution Summary

HKL-VITS contributes:

## 1. Hybrid Linguistic Architecture

Combines:

* rule based NLP
* Transformer
* BiLSTM
* Attention
* VITS

---

## 2. Kannada Phonological Modeling

Handles:

* vowel length
* gemination
* conjunct letters
* Sanskrit-origin words

---

## 3. Prosody Enhancement

Predicts:

* pause
* pitch
* duration
* energy

---

## 4. Emotion-Aware Kannada Speech

Supports:

```
neutral

happy

sad

angry

questioning
```

---

# 24. Final HKL-VITS Definition

HKL-VITS is a hybrid Kannada Text-to-Speech architecture that combines:

* Kannada linguistic preprocessing
* custom G2P conversion
* Transformer-based grapheme understanding
* BiLSTM phoneme modeling
* attention-based feature fusion
* prosody prediction
* emotion conditioning
* fine-tuned MMS-VITS acoustic synthesis

The goal is not to replace VITS, but to enhance a powerful speech synthesis backbone with Kannada-specific intelligence.

The final system provides:

* better pronunciation
* better timing
* better emotional expression
* better handling of Kannada linguistic complexity

---

## Final Architecture Statement

**HKL-VITS = Kannada Linguistic Intelligence + Neural Speech Generation**

or:

```
Language Understanding
        +
Speech Intelligence
        +
VITS Generation

        =

High Quality Kannada TTS System
```

**End of Part 3 — Complete HKL-VITS Implementation Report**
