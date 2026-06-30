# HKL-VITS Implementation Details Report

# Part 2 — Integrated Model Architecture, Neural Components, Implementation Details, and Data Flow

---

# 4. HKL-VITS Complete Hybrid Architecture

## 4.1 Overview

HKL-VITS (Hybrid Kannada Linguistic Enhanced VITS) is a hybrid Text-to-Speech architecture where different components solve different parts of Kannada speech generation.

A traditional TTS system:

```
Text
 |
Text Encoder
 |
VITS
 |
Speech
```

The encoder must learn everything:

* Kannada spelling rules
* pronunciation
* consonant clusters
* vowel changes
* rhythm
* pauses
* speech generation

This is difficult because Kannada has complex linguistic rules.

HKL-VITS separates the responsibilities:

```
                 Kannada Text
                      |
                      |
          Text Normalization Layer
                      |
                      |
        --------------------------------
        |                              |
        |                              |
 Grapheme Representation        G2P Conversion
        |                              |
        |                              |
        |                        Phoneme Tokens
        |                              |
        |                              |
 Transformer Encoder             BiLSTM Encoder
        |                              |
        |                              |
        --------------------------------
                      |
             Fusion Attention
                      |
                      |
          Hybrid Kannada Features
                      |
                      |
          Prosody Prediction Model
                      |
        --------------------------------
        |              |               |
      Pitch        Duration        Energy
                      |
                      |
              MMS-VITS Backbone
                      |
                      |
              Generated Waveform
                      |
                      |
              Kannada Speech Audio
```

---

# 5. Complete Data Flow in Current Implementation

The actual implementation pipeline:

```
Input Text

"ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ₹250 ಇದೆ"

        |
        ↓

KannadaTextNormalizer

"ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ರೂಪಾಯಿ ಎರಡು ನೂರು ಐವತ್ತು ಇದೆ"

        |
        ↓

HKLKannadaG2P

"ni mːa kha te ye lli ru paa yi..."

        |
        ↓

Two Representations


1. Grapheme

Original Kannada characters

        |
        ↓

KannadaGraphemeEncoder


2. Phoneme

Pronunciation sequence

        |
        ↓

KannadaPhonemeEncoder


        |
        ↓

HKLFusionAttention


        |
        ↓

KannadaProsodyModel


        |
        ↓

MMS-VITS


        |
        ↓

Waveform

```

---

# 6. Module 1 — Text Normalization Layer

## Purpose

Humans write text in a compact format.

Speech models require pronunciation-ready text.

Example:

Input:

```
ನನ್ನ ಬಳಿ ₹250 ಇದೆ
```

The model should not learn:

```
₹
250
```

directly.

It should receive:

```
ನನ್ನ ಬಳಿ ರೂಪಾಯಿ ಎರಡು ನೂರು ಐವತ್ತು ಇದೆ
```

---

# Implementation

Class:

```python
class KannadaTextNormalizer
```

File responsibility:

```
Text
 ↓
Cleaning
 ↓
Symbol expansion
 ↓
Number expansion
 ↓
Punctuation handling
```

---

## Number Expansion

Implementation:

```python
num2words(value, lang="kn")
```

Library:

```
indic_numtowords
```

Example:

```python
250
```

becomes:

```
ಎರಡು ನೂರು ಐವತ್ತು
```

---

## Symbol Expansion

Configuration:

```python
symbol_map
```

Example:

```python
"₹": "ರೂಪಾಯಿ"
"%": "ಶೇಕಡಾ"
"+": "ಪ್ಲಸ್"
```

Input:

```
50%
```

Output:

```
ಐವತ್ತು ಶೇಕಡಾ
```

---

## Why Rule Based?

Numbers and symbols are deterministic.

The neural model should not waste learning:

```
250 = two hundred fifty
```

This should be solved before training.

---

# 7. Module 2 — Kannada G2P Engine

## What is G2P?

G2P means:

```
Grapheme
(Text character)

        ↓

Phoneme
(Speech sound)
```

Example:

Kannada:

```
ಕನ್ನಡ
```

Characters:

```
ಕ ಣ್ ಣ ಡ
```

Sound:

```
ka nː na ɖa
```

---

## Why Kannada Needs G2P?

Because Kannada spelling and pronunciation are not always equal.

Examples:

## Consonant length

```
ಕ
```

short sound

```
ಕ್ಕ
```

long consonant

Speech difference:

```
ka

kːa
```

---

## Conjunct characters

Example:

```
ಕ್ಷ
```

A character model may see:

```
ಕ + ಷ
```

but speech requires:

```
ksha
```

---

# Current Implementation

Class:

```python
HKLKannadaG2P
```

Main functions:

```python
convert_word()
convert_sentence()
to_phoneme_string()
```

Example:

```python
g2p.convert_sentence("ಕನ್ನಡ")
```

Output:

```
ka nː na ɖa
```

---

# 8. Module 3 — Grapheme Encoder

## What is a Grapheme Encoder?

Grapheme means:

```
written form
```

Example:

```
ಕನ್ನಡ
```

The grapheme encoder learns:

* character relationships
* spelling structure
* Kannada script patterns

---

# Technology Used

Implementation:

```python
nn.TransformerEncoder
```

Library:

```
PyTorch
torch.nn
```

---

# What is Transformer?

A Transformer is a neural network that understands relationships between tokens.

Traditional models:

```
ಕ → ಣ → ಡ
```

read one by one.

Transformer:

```
ಕ ↔ ಣ ↔ ಡ
```

looks at the complete sequence.

---

Example:

Word:

```
ಮನೆಗಳಿಂದ
```

Contains:

```
ಮನೆ + ಗಳ + ಇಂದ
```

The ending:

```
ಇಂದ
```

depends on the beginning:

```
ಮನೆ
```

Attention connects these parts.

---

# Current Implementation

Class:

```python
KannadaGraphemeEncoder
```

Code:

```python
self.embed = nn.Embedding(
    vocab,
    hidden
)
```

Embedding converts:

```
Kannada character ID

        ↓

256 dimensional vector
```

Example:

Before:

```
ಕ = 45
```

After:

```
[
0.21,
0.54,
...
256 values
]
```

---

Transformer:

```python
nn.TransformerEncoderLayer(
    hidden,
    4,
    batch_first=True
)
```

Parameters:

| Parameter | Value | Meaning            |
| --------- | ----- | ------------------ |
| hidden    | 256   | feature size       |
| heads     | 4     | attention groups   |
| layers    | 4     | transformer blocks |

Output:

```
(batch, text_length, 256)
```

Example:

```
(2,50,256)
```

Meaning:

```
2 sentences
50 characters
256 features
```

---

# 9. Module 4 — Phoneme Encoder (BiLSTM)

## What is LSTM?

LSTM:

Long Short-Term Memory

It is a neural network designed for sequences.

Speech is sequential:

```
ka → n → na → ɖa
```

Every sound depends on previous and future sounds.

---

# Problem with Normal RNN

Long sequences:

```
ಮನೆಗಳಿಂದ
```

contain:

```
ಮನೆ + ಗಳ + ಇಂದ
```

A simple RNN may forget:

```
ಮನೆ
```

when reaching:

```
ಇಂದ
```

---

# LSTM Solution

LSTM has memory.

It uses three gates.

---

## Forget Gate

Question:

"What old information should I remove?"

Example:

Remove irrelevant pronunciation history.

---

## Input Gate

Question:

"What new information should I remember?"

Example:

Store suffix information.

---

## Output Gate

Question:

"What information should continue?"

---

# What is BiLSTM?

BiLSTM =

```
Forward LSTM
+
Backward LSTM
```

---

Forward direction:

```
ka → nː → na → ɖa
```

Learns:

"What came before?"

---

Backward direction:

```
ɖa → na → nː → ka
```

Learns:

"What comes after?"

---

Final:

```
Forward hidden

+

Backward hidden

=

Complete phoneme understanding
```

---

# Current Implementation

Class:

```python
KannadaPhonemeEncoder
```

Code:

```python
nn.LSTM(
 hidden,
 hidden//2,
 num_layers=2,
 bidirectional=True
)
```

Parameters:

| Parameter   | Value         |
| ----------- | ------------- |
| input size  | 256           |
| hidden size | 128           |
| layers      | 2             |
| direction   | bidirectional |

Why hidden//2?

Because:

Forward:

```
128
```

Backward:

```
128
```

Combined:

```
256
```

which matches the grapheme encoder.

---

Output:

```
(batch, phoneme_length,256)
```

---

# 10. Module 5 — Fusion Attention Layer

Now we have:

Grapheme knowledge:

```
How word is written
```

Phoneme knowledge:

```
How word sounds
```

Need to combine them.

---

Simple addition:

```
grapheme + phoneme
```

does not know importance.

---

Attention solves this.

---

# What is Attention?

Attention asks:

"Which information is more important here?"

Example:

Word:

```
ಟೆಕ್ನಾಲಜಿ
```

Need spelling:

```
Grapheme importance ↑
```

Word:

```
ಕನ್ನಡ
```

Need pronunciation:

```
Phoneme importance ↑
```

---

# Current Implementation

Class:

```python
HKLFusionAttention
```

Code:

```python
nn.MultiheadAttention(
256,
4
)
```

Parameters:

| Parameter       | Value |
| --------------- | ----- |
| hidden size     | 256   |
| attention heads | 4     |

---

Output:

```
Hybrid Kannada Feature

(batch,sequence,256)
```

---

# 11. Module 6 — Prosody Prediction Model

Correct pronunciation is not enough.

Human speech needs:

* pause
* speed
* pitch
* energy

Example:

Same text:

```
ನೀನು ಬಂದೆ
```

Statement:

```
ನೀನು ಬಂದೆ.
```

Question:

```
ನೀನು ಬಂದೆ?
```

Different:

* pitch
* ending
* pause

---

# Current Implementation

Class:

```python
KannadaProsodyModel
```

Architecture:

```
HKL Feature

     ↓

BiLSTM

     ↓

Linear Layers

     ↓

Pause
Duration
Pitch
Energy

```

---

Parameters:

```python
hidden_dim=256

prosody_dim=4
```

Output:

```
pause
duration
pitch
energy
```

---

# 12. MMS-VITS Backbone Integration

## What is MMS-VITS?

MMS-VITS is a multilingual VITS speech generation model from HuggingFace.

Library:

```
transformers
```

Implementation:

```python
VitsModel.from_pretrained()
```

Model:

```
facebook/mms-tts-kan
```

---

# Role of MMS-VITS

HKL modules understand Kannada language.

MMS-VITS converts:

```
linguistic features

        ↓

audio waveform
```

---

# Freezing Backbone

Configuration:

```python
freeze_mms=True
```

Meaning:

MMS weights are not updated.

Only HKL layers learn.

Advantages:

* less GPU memory
* faster training
* preserves pretrained speech quality

---

# 13. Complete Training Flow

```
Dataset

WAV + Text

      ↓

HKLDataset

      ↓

Tokenizer

      ↓

Grapheme Encoder

      ↓

Phoneme Encoder

      ↓

Fusion Attention

      ↓

Prosody Model

      ↓

Loss

      ↓

AdamW Optimizer

      ↓

Checkpoint
```

---

# 14. Current Implementation Validation

Your API test confirms the pipeline is working:

Example:

Input:

```
ನಿಮ್ಮ ಉಳಿತಾಯ ಖಾತೆಯಲ್ಲಿ 2500 ರೂಪಾಯಿಗಳು
```

Normalization:

```
ನಿಮ್ಮ ಉಳಿತಾಯ ಖಾತೆಯಲ್ಲಿ ಎರಡು ಸಾವಿರದ ಐನೂರು ರೂಪಾಯಿಗಳು
```

Generated:

```
Samples: 230912
```

The complete chain is working:

```
Text

↓

Normalization

↓

G2P

↓

HKL Features

↓

MMS-VITS

↓

Waveform

```

---

# End of Part 2
