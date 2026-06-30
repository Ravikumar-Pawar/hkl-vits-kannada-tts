# HKL-VITS: Hybrid Kannada Linguistic Enhanced VITS

# Part 1 — Complete Architecture Overview, Motivation, Implementation Design, and Integrated Model Explanation

---

# 1. Introduction

HKL-VITS (Hybrid Kannada Linguistic Enhanced VITS) is a Kannada Text-to-Speech (TTS) research architecture that improves Kannada speech generation by adding language-specific processing layers before the existing MMS-VITS speech generation backbone.

The main idea is:

A general TTS model learns speech generation, but Kannada pronunciation contains many language-specific rules that can be explicitly modeled.

Instead of depending only on a single text encoder:

```text
Text
 |
Tokenizer
 |
TTS Model
 |
Audio
```

HKL-VITS introduces additional Kannada linguistic processing:

```text
Kannada Text
        |
        ↓
Text Normalization
        |
        ↓
Kannada G2P
        |
        ↓
Dual Linguistic Representation
        |
        ├────────────────┐
        ↓                ↓
Grapheme Encoder    Phoneme Encoder
        ↓                ↓
        └──── Fusion Attention ────┘
                     |
                     ↓
          Kannada Linguistic Features
                     |
                     ↓
             Prosody Prediction
                     |
                     ↓
              MMS-VITS Backbone
                     |
                     ↓
              Audio Waveform
```

HKL-VITS separates different responsibilities:

* Rule-based modules handle deterministic Kannada processing.
* G2P provides pronunciation information.
* Neural encoders learn linguistic patterns.
* Prosody model learns speech characteristics.
* MMS-VITS generates the final waveform.

---

# 2. Motivation — Why Kannada Needs Linguistic Enhancement

Kannada is an abugida script where written characters do not always directly represent complete pronunciation units.

Examples:

## Vowel changes

```
ಕ

ka
```

```
ಕಾ

kaa
```

```
ಕಿ

ki
```

```
ಕೀ

kii
```

The base consonant changes pronunciation depending on vowel markers.

---

## Consonant combinations

Kannada contains conjunct characters:

Example:

```
ಕ್ಷ
```

This represents:

```
ksha
```

A normal character-based model must learn this relationship from data.

HKL-VITS provides explicit pronunciation information.

---

# 3. HKL-VITS Architecture Components

Current implementation consists of these modules:

| Module                  | Implementation       | Purpose                                     |
| ----------------------- | -------------------- | ------------------------------------------- |
| Kannada Text Normalizer | Python rule engine   | Converts raw text into speech-friendly text |
| Number Expansion        | indic_numtowords     | Converts numbers into Kannada words         |
| Symbol Expansion        | Rule mapping         | Converts symbols into spoken forms          |
| Kannada G2P             | Rule-based mapper    | Converts Kannada script into phoneme tokens |
| Grapheme Encoder        | Transformer Encoder  | Learns written Kannada representation       |
| Phoneme Encoder         | BiLSTM               | Learns pronunciation sequence               |
| Fusion Attention        | Multi-head Attention | Combines linguistic representations         |
| Prosody Model           | BiLSTM network       | Predicts pause, duration, pitch, energy     |
| MMS-VITS                | facebook/mms-tts-kan | Generates speech waveform                   |
| Trainer Wrapper         | PyTorch model        | Trains HKL linguistic components            |

---

# 4. Text Normalization Layer

Implementation:

```python
KannadaTextNormalizer
```

Purpose:

Convert human-written text into a form suitable for speech generation.

Example:

Input:

```
ನನ್ನ ಖಾತೆಯಲ್ಲಿ ₹2500 ಇದೆ
```

Processing:

Currency expansion:

```
₹2500
```

becomes:

```
ರೂಪಾಯಿ 2500
```

Number expansion:

```
2500
```

becomes:

```
ಎರಡು ಸಾವಿರದ ಐನೂರು
```

Final normalized text:

```
ನನ್ನ ಖಾತೆಯಲ್ಲಿ ರೂಪಾಯಿ ಎರಡು ಸಾವಿರದ ಐನೂರು ಇದೆ
```

---

# 4.1 Number Expansion

Implemented using:

```python
indic_numtowords
```

Example:

Input:

```
2026
```

Output:

```
ಎರಡು ಸಾವಿರ ಇಪ್ಪತ್ತಾರು
```

Another example:

Input:

```
23,423
```

Output:

```
ಇಪ್ಪತ್ತ್ಮೂರು ಸಾವಿರದ ನಾನ್ನೂರಿಪ್ಪತ್ತ್ಮೂರು
```

The purpose is to prevent the speech model from trying to pronounce numeric symbols directly.

---

# 4.2 Symbol Expansion

Configured symbols:

```text
₹  → ರೂಪಾಯಿ

$  → ಡಾಲರ್

%  → ಶೇಕಡಾ

+  → ಪ್ಲಸ್

=  → ಸಮಾನ

&  → ಮತ್ತು
```

Example:

Input:

```
A+B=C
```

Normalized:

```
A ಪ್ಲಸ್ B ಸಮಾನ C
```

---

# 4.3 Punctuation Processing

Punctuation is converted into speech pause information.

Example:

Input:

```
ನೀವು ಹೇಗಿದ್ದೀರಿ?
```

After normalization:

```
ನೀವು ಹೇಗಿದ್ದೀರಿ ...
```

This helps create natural pauses.

---

# 5. Kannada G2P Module

Implementation:

```python
HKLKannadaG2P
```

Purpose:

Convert Kannada written characters into pronunciation units.

Example:

Input:

```
ಕ
```

Output:

```
ka
```

---

Input:

```
ಕ್ಷ
```

Output:

```
ksha
```

---

Supported conjuncts:

```text
ಕ್ಷ  → ksha

ಪ್ರ  → pra

ಕ್ರ  → kra

ಗ್ರ  → gra

ತ್ರ  → tra
```

---

Example:

Input:

```
ಕನ್ನಡ
```

Approximate phoneme representation:

```
ka ṇa na ɖa
```

The phoneme representation gives the model pronunciation information.

---

# 6. Grapheme Encoder

Implementation:

```python
KannadaGraphemeEncoder
```

Technology:

```
Transformer Encoder
```

Input:

Kannada character IDs

Output:

```
[batch, sequence, 256]
```

Purpose:

Learn written Kannada structure.

It captures:

* character relationships
* spelling patterns
* word structure
* context

Example:

Input:

```
ಮನೆಗೆ
```

The encoder understands:

```
ಮನೆ + ಗೆ
```

as a connected linguistic structure.

---

# 6.1 Transformer Attention Concept

Traditional sequence processing:

```
ಕ → ನ → ಡ
```

Processes characters individually.

Transformer attention learns relationships:

```
ಕ ↔ ನ ↔ ಡ
```

The model can use information from the complete word.

---

# 7. Phoneme Encoder

Implementation:

```python
KannadaPhonemeEncoder
```

Technology:

```
Bidirectional LSTM
```

Input:

Phoneme sequence:

```
ma ne
```

Output:

```
[batch, sequence, 256]
```

---

# Why BiLSTM?

Pronunciation depends on surrounding sounds.

Example:

```
ಅಣ್ಣ
```

contains consonant doubling.

The pronunciation depends on:

Previous sound:

```
ಅ
```

Current sound:

```
ಣ್ಣ
```

Future context:

```
word ending
```

A BiLSTM reads:

Forward:

```
sound → sound → sound
```

Backward:

```
sound ← sound ← sound
```

The output contains complete phoneme context.

---

# 8. Dual Linguistic Representation

HKL-VITS uses two representations.

## Grapheme representation

Written Kannada:

```
ಕನ್ನಡ
```

Information:

* spelling
* character structure
* word formation

---

## Phoneme representation

Pronunciation:

```
ka ṇa na ɖa
```

Information:

* speech sound
* pronunciation sequence

---

Combined architecture:

```text
                 Kannada Text

                      |
        --------------------------------

        ↓                              ↓

 Grapheme Encoder              Phoneme Encoder


 Written form                  Sound form


        ↓                              ↓

        --------------------------------

                      ↓

             Fusion Attention

                      ↓

          Kannada Linguistic Features
```

---

# 9. Fusion Attention Layer

Implementation:

```python
HKLFusionAttention
```

Technology:

```
Multi-head Attention
```

Input:

Grapheme features:

```
[batch, sequence, 256]
```

Phoneme features:

```
[batch, sequence, 256]
```

The attention layer learns how to combine:

* written information
* pronunciation information

Output:

```
Unified Kannada linguistic representation
```

---

# 10. Prosody Prediction Model

Implementation:

```python
KannadaProsodyModel
```

Technology:

```
Bidirectional LSTM
```

Purpose:

Predict speech characteristics.

The model produces:

```text
Pause

Duration

Pitch

Energy
```

Example:

Sentence:

```
ನಮಸ್ಕಾರ, ನಾನು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇನೆ.
```

Prosody model helps represent:

* where pauses occur
* sound length
* pitch variation
* energy changes

---

# 11. MMS-VITS Integration

Backbone:

```python
facebook/mms-tts-kan
```

Role:

Generate waveform audio.

Flow:

```text
Normalized Kannada Text

        ↓

MMS Tokenizer

        ↓

MMS-VITS Model

        ↓

Waveform Tensor

        ↓

WAV Audio
```

The current implementation uses MMS-VITS as the speech generation backbone.

---

# 12. Real Inference Flow

Example input:

```
ನಿಮ್ಮ ಉಳಿತಾಯ ಖಾತೆಯಲ್ಲಿ ₹2500 ಜಮೆಯಾಗಿವೆ.
```

Step 1:

Normalization:

```
ನಿಮ್ಮ ಉಳಿತಾಯ ಖಾತೆಯಲ್ಲಿ
ಎರಡು ಸಾವಿರದ ಐನೂರು ರೂಪಾಯಿಗಳು ಜಮೆಯಾಗಿವೆ ...
```

Step 2:

Tokenization:

```
Kannada text → MMS token IDs
```

Step 3:

G2P:

```
Kannada text → phoneme sequence
```

Step 4:

HKL linguistic processing:

```
Grapheme Encoder
+
Phoneme Encoder
+
Fusion Attention
```

Step 5:

MMS-VITS:

```
Text representation
        ↓
Waveform
```

Output:

```
Kannada speech audio
```

---

# 13. Verified API Testing

## Test 1

Input:

```
23423
```

Normalization:

```
ಇಪ್ಪತ್ತ್ಮೂರು ಸಾವಿರದ ನಾನ್ನೂರಿಪ್ಪತ್ತ್ಮೂರು
```

Result:

```
Synthesis completed
Samples: 59904
```

---

## Test 2

Input:

```
ನಿಮ್ಮ ಉಳಿತಾಯ ಖಾತೆಯಲ್ಲಿ 2500 ರೂಪಾಯಿಗಳು ಜಮೆಯಾಗಿವೆ.
```

Normalization:

```
ನಿಮ್ಮ ಉಳಿತಾಯ ಖಾತೆಯಲ್ಲಿ ಎರಡು ಸಾವಿರದ ಐನೂರು ರೂಪಾಯಿಗಳು ಜಮೆಯಾಗಿವೆ ...
```

Result:

```
Synthesis completed
Samples: 230912
```

---

# 14. Final HKL-VITS Architecture Summary

```text
Raw Kannada Text

        ↓

Text Normalization

        ↓

Kannada G2P

        ↓

Dual Linguistic Encoding

        ↓

Transformer Grapheme Encoder

        +

BiLSTM Phoneme Encoder

        ↓

Fusion Attention

        ↓

Prosody Prediction

        ↓

MMS-VITS Backbone

        ↓

Kannada Speech Waveform
```

HKL-VITS combines:

* deterministic Kannada rules
* pronunciation modeling
* linguistic representation learning
* prosody modeling
* neural speech synthesis


