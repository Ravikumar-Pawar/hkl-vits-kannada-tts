# HKL-VITS Implementation Details Report

## Part 2 — Detailed Architecture, Integrated Models, Data Flow, and Module Responsibilities

## 4. HKL-VITS Complete Hybrid Architecture

HKL-VITS (Hybrid Kannada Linguistic-enhanced VITS) is not a single neural network. It is a combination of multiple specialized models where each model solves a specific problem in Kannada Text-to-Speech.

A normal TTS model tries to learn everything:

```
Text
 ↓
Single Encoder
 ↓
VITS
 ↓
Audio
```

The problem is that Kannada has many linguistic challenges:

* Same character can have different pronunciations
* Vowel length changes meaning
* Consonant doubling changes pronunciation
* Compound words become very long
* Numbers and symbols are difficult
* Emotional expression requires correct pauses and pitch

HKL-VITS separates these problems into different intelligent modules.

The complete architecture:

```
                 Kannada Input Text
                         |
                         |
             Text Normalization Layer
                         |
        --------------------------------
        |                              |
 Grapheme Processing              G2P Conversion
        |                              |
        |                         Phoneme Sequence
        |                              |
        |                              |
        |                    Phoneme Encoder
        |                         (BiLSTM)
        |                              |
        |
 Grapheme Encoder
 (Transformer)
        |
        |
        -------------------------------
                         |
                  Fusion Attention
                         |
             Hybrid Linguistic Feature
                         |
                         |
              Prosody Prediction Model
                   (BiLSTM)
                         |
          --------------------------------
          |              |               |
        Pitch        Duration        Energy
          |
          |
      Emotion Embedding
          |
          |
       MMS-VITS Backbone
       (Fine-tuned Kannada Model)
          |
          |
      Neural Vocoder
          |
          |
        Waveform
          |
          |
    Audio Post Processing
          |
          |
       Final Kannada Speech
```

---

# 5. Module 1 — Text Normalization Layer

## Purpose

The first problem in TTS is that humans write text in a compact form, but speech requires expanded pronunciation.

Example:

Input:

```
ನಾನು 123 ಪುಸ್ತಕಗಳನ್ನು ಓದಿದೆ
```

A human reads:

```
ನಾನು ಒಂದು ಎರಡು ಮೂರು ಪುಸ್ತಕಗಳನ್ನು ಓದಿದೆ
```

The model should not learn every possible number combination.

Therefore HKL-VITS uses a deterministic preprocessing layer.

---

## Responsibilities

### 1. Number Expansion

Digits:

```
1 → ಒಂದು
2 → ಎರಡು
3 → ಮೂರು
```

Example:

Input:

```
ನನ್ನ ಬಳಿ 25 ರೂಪಾಯಿ ಇದೆ
```

Converted:

```
ನನ್ನ ಬಳಿ ಎರಡು ಐದು ರೂಪಾಯಿ ಇದೆ
```

---

### 2. Symbol Expansion

Symbols do not have natural pronunciation.

Example:

Input:

```
A+B=C
```

Normalization:

```
A ಪ್ಲಸ್ B ಸಮಾನ C
```

Mapping:

```
+
ಪ್ಲಸ್

=
ಸಮಾನ

%
ಶೇಕಡಾ

&
ಮತ್ತು
```

---

### 3. Punctuation Preservation

Punctuation contains speech information.

Example:

Without punctuation:

```
ನೀನು ಬಂದೆಯಾ
```

With question:

```
ನೀನು ಬಂದೆಯಾ?
```

The second one requires:

* higher pitch ending
* different intonation

HKL-VITS keeps punctuation information for prosody prediction.

---

# 6. Module 2 — Kannada G2P Engine

## Grapheme To Phoneme Conversion

G2P converts written Kannada characters into pronunciation units.

Example:

Input:

```
ಕನ್ನಡ
```

Grapheme:

```
ಕ ಣ್ ಣ ಡ
```

Phoneme:

```
ka nː na ɖa
```

The neural model receives pronunciation information instead of only characters.

---

# Why G2P is Important for Kannada?

English:

```
cat
```

Almost always:

```
k æ t
```

Kannada:

```
ಕ
```

can have different contexts.

Also Kannada has:

## Vowel Length

Short:

```
ಕಿ
```

Long:

```
ಕೀ
```

They are different sounds.

Representation:

```
i
i:
```

---

## Gemination

Single consonant:

```
ಕ
```

Double consonant:

```
ಕ್ಕ
```

Pronunciation:

```
ka
kːa
```

The duration changes.

---

## Consonant Clusters

Example:

```
ಕ್ಷ
```

Not:

```
ka + sha
```

Actual:

```
ksha
```

---

HKL G2P handles:

```
ಕನ್ನಡ
→ ka nː na ɖa


ಅಣ್ಣ
→ a ṇː ṇa


ಪ್ರ
→ pra


ಕ್ಷ
→ ksha


ಕ್ರ
→ kra
```

---

# 7. Module 3 — Grapheme Encoder

## Technology

Transformer Encoder

---

## Purpose

The grapheme encoder understands the original Kannada writing structure.

Input:

```
ಕನ್ನಡ
```

Character sequence:

```
ಕ ಣ್ ಣ ಡ
```

The encoder learns:

* spelling pattern
* character relationships
* script structure
* visual similarity

---

## Why Transformer?

Kannada words can be long.

Example:

```
ಮನೆಗಳಿಂದ
```

Contains:

```
ಮನೆ + ಗಳ + ಇಂದ
```

The meaning depends on relationships between distant characters.

Transformer uses attention:

Example:

```
ಮನೆಗಳಿಂದ
```

The character:

```
ಇಂದ
```

depends on:

```
ಮನೆ
```

Attention connects them.

---

Transformer gives:

```
Grapheme Feature

(batch, sequence, hidden)

Example:

[2, 50, 256]
```

Meaning:

2 sentences

50 tokens

256 feature dimensions

---

# 8. Module 4 — Phoneme Encoder

## Technology

Bi-directional LSTM (BiLSTM)

---

## Why BiLSTM?

Speech pronunciation is sequential.

Example:

```
ಕನ್ನಡ
```

Pronunciation:

```
ka nː na ɖa
```

Each sound affects the next sound.

A normal LSTM reads:

```
ka → nː → na → ɖa
```

Only previous information is available.

A BiLSTM reads both directions.

Forward:

```
ka → nː → na → ɖa
```

Backward:

```
ɖa → na → nː → ka
```

So each phoneme receives:

* previous context
* future context

---

## BiLSTM Internal Working

A BiLSTM has two LSTMs.

### Forward LSTM

Input:

```
k a n n a
```

Learning:

```
what came before
```

Example:

While processing:

```
na
```

it knows:

```
ka
```

came earlier.

---

### Backward LSTM

Reads:

```
a n n a k
```

Learning:

```
what comes after
```

Example:

While processing:

```
na
```

it knows:

```
ɖa
```

comes later.

---

The outputs are combined:

```
Forward Hidden
        +
Backward Hidden
        |
        |
Combined Phoneme Feature
```

---

## Why BiLSTM instead of Transformer for phoneme?

Transformer is excellent for long text relationships.

But phoneme sequences require:

* pronunciation continuity
* local sound transitions
* timing relationships

BiLSTM naturally captures speech sequence behavior.

---

Example:

Without BiLSTM:

```
ka n na
```

may sound:

```
ka-na-na
```

With BiLSTM:

```
ka-nːa
```

because it understands:

```
double consonant duration
```

---

# 9. Module 5 — Fusion Attention Layer

Now HKL-VITS has two different knowledge sources.

## Grapheme Encoder knows:

```
How the word is written
```

## Phoneme Encoder knows:

```
How the word should sound
```

Both are important.

---

Simple combination:

```
grapheme + phoneme
```

is not enough.

Because sometimes:

Grapheme is more reliable.

Example:

technical words

```
AI
```

Need spelling information.

Sometimes phoneme is more reliable.

Example:

```
ಕನ್ನಡ
```

Pronunciation is more important.

---

Therefore HKL-VITS uses:

## Multi-Head Attention Fusion

The model learns:

```
For this position:

70% phoneme
30% grapheme
```

or

```
40% phoneme
60% grapheme
```

depending on the word.

---

Output:

```
Hybrid Linguistic Representation
```

Example:

Before:

```
Grapheme:

[batch,49,256]


Phoneme:

[batch,49,256]
```

After fusion:

```
HKL Feature:

[batch,49,256]
```

This becomes the main linguistic representation.

---

# 10. Module 6 — Prosody Prediction Network

## Purpose

Correct words are not enough.

Speech requires:

* where to pause
* how long to speak
* pitch movement
* energy

Example:

Same sentence:

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

Different prosody.

---

HKL-VITS predicts:

```
Pause
Duration
Pitch
Energy
```

---

Architecture:

```
HKL Features

      |
      |

BiLSTM Prosody Encoder

      |
      |

Linear Prediction Heads

      |
 -----------------
 |       |        |
Pause Duration Pitch Energy
```

---

Output:

```
pause:

[batch, tokens]


duration:

[batch,tokens]


pitch:

[batch,tokens]


energy:

[batch,tokens]
```

---

This information controls MMS-VITS synthesis.

---

# 11. Module 7 — Emotion Embedding

Optional module.

Supported:

```
neutral
happy
sad
angry
questioning
```

Emotion vector is added:

Example:

Neutral:

```
linguistic feature
+
neutral embedding
```

Happy:

```
linguistic feature
+
happy embedding
```

The model changes:

* pitch
* energy
* speaking style

---

# End of Part 2

Part 3 will cover:

* MMS-VITS fine-tuned backbone integration
* Training pipeline
* Loss functions
* Parameter details
* Colab T4 implementation
* Inference flow
* Comparison with existing TTS models
* Why HKL-VITS is different academically and technically
