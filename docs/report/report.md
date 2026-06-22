# HKL-VITS Implementation Details

## Hybrid Linguistic-Enhanced VITS for Kannada Text-to-Speech

**Project Type:** Research Prototype
**Language:** Kannada (ಕನ್ನಡ)
**Base Model:** facebook/mms-tts-kan (MMS VITS Kannada)
**Framework:** PyTorch + Hugging Face Transformers
**Hardware Tested:** NVIDIA Tesla T4 GPU (Colab)
**Architecture:** Hybrid Linguistic Frontend + Prosody Modeling + VITS Acoustic Backbone

---

# 1. Overview

HKL-VITS (Hybrid Kannada Linguistic VITS) is a Kannada-specific neural Text-to-Speech architecture designed to improve pronunciation accuracy, linguistic understanding, and expressive speech generation by combining multiple specialized modules.

Instead of training a complete TTS model from scratch, HKL-VITS integrates a pretrained VITS synthesis model with Kannada-aware linguistic modules.

The core idea:

> Use different models for different problems.

A single TTS model struggles with:

* Kannada morphology
* vowel length
* consonant gemination
* conjunct consonants
* numbers and symbols
* emotional phrasing
* pronunciation of rare words

HKL-VITS solves these using a hybrid pipeline.

---

# 2. Overall HKL-VITS Architecture

```
                    Kannada Text Input

                           |
                           ↓

             Text Normalization Layer

        Numbers
        Symbols
        Punctuation
        Abbreviations

                           |
                           ↓

              Kannada G2P Engine

        Grapheme → Phoneme Conversion

        Example:

        ಕನ್ನಡ

        ka nː na ɖa


                           |
             ┌─────────────┴─────────────┐

             ↓                           ↓

   Grapheme Encoder              Phoneme Encoder

   Transformer                   BiLSTM

   Learns:                       Learns:

   - Kannada script              - Pronunciation
   - Character patterns          - Gemination
   - Context                     - Vowel length
                                 - Clusters


             └─────────────┬─────────────┘

                           ↓


              Fusion Attention Layer


                           ↓


             Kannada Linguistic Feature


                    [Batch, Tokens, 256]


                           ↓


              Prosody Prediction Model


          ┌─────────┬────────┬────────┐

          ↓         ↓        ↓        ↓

       Pause    Duration   Pitch   Energy


                           ↓


              Emotion Conditioning

          neutral
          happy
          sad
          angry
          questioning


                           ↓


              MMS-VITS Backbone


             Text → Mel → Waveform


                           ↓


              Audio Post Processing


                           ↓


              Kannada Speech Output

```

---

# 3. Module Implementation Details

---

# 3.1 Configuration Layer

The complete system is controlled using HKLConfig.

Example:

```python
@dataclass
class HKLConfig:

    mms_model_name = "facebook/mms-tts-kan"

    sample_rate = 16000


    # Linguistic modules

    use_g2p = True

    use_grapheme_encoder = True

    use_phoneme_encoder = True


    # Prosody

    use_prosody_model = True


    prosody_features = [

        "pause",
        "duration",
        "pitch",
        "energy"

    ]


    # Emotion

    use_emotion_embedding = True


    emotions = [

        "neutral",
        "happy",
        "sad",
        "angry",
        "questioning"

    ]
```

---

# 4. Text Normalization Module

## Purpose

Convert raw Kannada text into speech-friendly text.

Handles:

* numbers
* symbols
* punctuation
* abbreviations

Example:

Input:

```
ನಾನು 123 ಮಾತನಾಡುತ್ತೇನೆ
```

Output:

```
ನಾನು ಒಂದು ಎರಡು ಮೂರು ಮಾತನಾಡುತ್ತೇನೆ
```

---

Example:

Input:

```
ಇದು 50% ಸರಿಯಾಗಿದೆ
```

Output:

```
ಇದು ಐದು ಸೊನ್ನೆ ಶೇಕಡಾ ಸರಿಯಾಗಿದೆ
```

Symbol handling:

```
+
ಪ್ಲಸ್


=
ಸಮಾನ


%
ಶೇಕಡಾ
```

Implementation:

```python
class KannadaNormalizer:


    def expand_numbers(self,text):

        ...


    def expand_symbols(self,text):

        ...


    def normalize(self,text):

        text = self.expand_numbers(text)

        text = self.expand_symbols(text)

        return text

```

---

# 5. Kannada G2P Module

## Purpose

Convert Kannada characters into pronunciation units.

Traditional TTS:

```
Text
 |
Characters
 |
Speech
```

Problem:

Kannada script does not always directly represent pronunciation.

HKL:

```
Text

↓

G2P

↓

Phonemes

↓

Speech
```

---

Examples:

## Vowel length

```
ಕಾರು

kaa ru
```

Long vowel preserved.

---

## Gemination

```
ಅಣ್ಣ

a ṇː ṇa

```

Double consonant duration preserved.

---

## Conjunct handling

```
ಪ್ರ

pra


ಕ್ಷ

ksha


ಕ್ರ

kra

```

---

Implementation:

```python
class KannadaG2P:


    def convert(self,text):

        phonemes=[]


        for char in text:

            phonemes.append(
                self.map_char(char)
            )


        return " ".join(phonemes)

```

---

# 6. Dual Encoder Architecture

This is the main linguistic innovation.

---

# 6.1 Grapheme Encoder

Technology:

Transformer Encoder

Input:

```
ಕನ್ನಡ
```

Learns:

* spelling
* character relationships
* script structure

Output:

```
[batch, tokens,256]

```

---

Implementation:

```python
class GraphemeEncoder(nn.Module):


    def forward(self,input_ids):

        return self.transformer(
            input_ids
        )

```

---

# 6.2 Phoneme Encoder

Technology:

BiLSTM

Input:

```
ka nː na ɖa
```

Learns:

* pronunciation
* sound transitions
* duration

Implementation:

```python
class PhonemeEncoder(nn.Module):


    def __init__(self):

        self.lstm = nn.LSTM(

            input_size=256,

            hidden_size=128,

            bidirectional=True

        )


    def forward(self,x):

        output,_ = self.lstm(x)

        return output

```

---

# 7. Fusion Layer

Instead of:

```
grapheme + phoneme
```

HKL uses attention.

Architecture:

```
Grapheme Feature

        +

Phoneme Feature

        |

Multi Head Attention

        |

Unified Kannada Representation

```

Implementation:

```python
class HKLFusion(nn.Module):


    def forward(
        self,
        grapheme,
        phoneme
    ):


        fused = self.attention(

            grapheme,

            phoneme,

            phoneme

        )


        return fused

```

Output:

```
torch.Size([1,49,256])
```

---

# 8. Prosody Model

Purpose:

Improve natural Kannada speaking style.

Predicts:

| Feature  | Meaning         |
| -------- | --------------- |
| Pause    | Sentence breaks |
| Duration | Speaking speed  |
| Pitch    | Voice melody    |
| Energy   | Loudness        |

Implementation:

```python
class ProsodyModel(nn.Module):


    def __init__(self):

        self.lstm = nn.LSTM(

            256,

            128,

            batch_first=True

        )


        self.output = nn.Linear(

            128,

            4

        )


    def forward(self,x):

        x,_ = self.lstm(x)

        return self.output(x)

```

Output:

```
[batch,tokens,4]


Example:

[
pause,
duration,
pitch,
energy

]

```

---

# 9. Emotion Module

Emotion embedding added before synthesis.

Supported:

```
neutral

happy

sad

angry

questioning

```

Implementation:

```python
class EmotionEmbedding(nn.Module):


    def __init__(self):

        self.embed = nn.Embedding(

            5,

            256

        )


    def forward(self,id):

        return self.embed(id)

```

Combination:

```
HKL features

+

Emotion vector

=

Emotion controlled features

```

---

# 10. MMS-VITS Integration

The acoustic generation is handled by MMS.

Input:

```
HKL representation

```

Adapter:

```
256 dimension

↓

192 dimension

```

Because MMS hidden size:

```
192
```

Implementation:

```python
class MMSAdapter(nn.Module):


    def __init__(self):

        self.linear = nn.Linear(

            256,

            192

        )


    def forward(self,x):

        return self.linear(x)

```

---

# 11. Final HKL-VITS Forward Flow

```python
def forward(text):


    text = normalizer(text)


    phonemes = g2p(text)


    g = grapheme_encoder(text)


    p = phoneme_encoder(phonemes)


    hkl = fusion(g,p)


    prosody = prosody_model(hkl)


    hkl = emotion(hkl)


    mms_input = adapter(hkl)


    audio = mms_model(

        input_ids=mms_input

    )


    return audio

```

---

# 12. Training Details

Dataset:

Kannada OpenSLR Dataset

Samples:

```
2214 utterances
```

Audio:

```
16kHz WAV
```

---

Training:

```
Epochs: 10

Batch size: 2

Gradient accumulation: 8

Learning rate:

0.0002

Optimizer:

AdamW

GPU:

Tesla T4
```

---

Trainable Parameters:

Before:

```
MMS frozen
```

HKL modules:

```
~2.6M parameters
```

Full model:

```
~44M parameters
```

---

# 13. Experimental Output

Training result:

```
Epoch 10 Loss:

0.0158

```

Inference:

```
HKL features:

[1,63,256]


Prosody:

[1,63,4]


Waveform:

generated successfully

```

---

# 14. Why HKL-VITS is Hybrid

Traditional TTS:

```
One model does everything

Text
 |
TTS
 |
Audio

```

HKL-VITS:

```
Rule Based NLP

+
Neural Linguistic Models

+
Prosody Network

+
Emotion Model

+
VITS Generator

```

Each module solves one problem.

---

# 15. Advantages

| Problem                    | Solution               |
| -------------------------- | ---------------------- |
| Kannada spelling ambiguity | Grapheme encoder       |
| Pronunciation errors       | G2P                    |
| Long vowels                | Phoneme representation |
| Gemination                 | Phoneme timing         |
| Numbers                    | Normalizer             |
| Expressionless voice       | Prosody model          |
| Emotion                    | Emotion embedding      |
| Training cost              | MMS transfer learning  |

---

# 16. Final Model Definition

HKL-VITS:

> A hybrid Kannada neural TTS architecture combining Kannada linguistic processing, dual text representation learning, prosody prediction, emotion conditioning, and pretrained MMS-VITS waveform synthesis.

The novelty is not replacing VITS.

The novelty is making VITS Kannada-aware.

---

# Final Architecture Summary

```
Kannada Text

↓

Normalization

↓

Kannada G2P

↓

Dual Encoder

(Grapheme Transformer)

+

(Phoneme BiLSTM)

↓

Attention Fusion

↓

Prosody + Emotion

↓

MMS-VITS

↓

Waveform

↓

Kannada Speech

```

This implementation provides a modular research architecture that can later extend to:

* multi-speaker Kannada TTS
* regional accents
* real-time streaming
* other Dravidian languages (Tamil/Telugu).
