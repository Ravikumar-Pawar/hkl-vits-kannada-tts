# HKL-VITS: Hybrid Kannada Linguistic Enhanced VITS

## Part 1 — Complete Architecture Overview, Motivation, and Integrated Model Explanation

## 1. Introduction

HKL-VITS (Hybrid Kannada Linguistic Enhanced VITS) is a Kannada Text-to-Speech (TTS) research architecture designed to generate natural Kannada speech by combining multiple specialized AI modules instead of depending on a single end-to-end model.

Traditional neural TTS systems generally follow:

```
Text Input
    ↓
Single Text Encoder
    ↓
TTS Model
    ↓
Audio Output
```

This approach works well for languages with simpler spelling-pronunciation relationships, but Kannada introduces several linguistic challenges:

* Complex consonant conjuncts
* Vowel length differences
* Gemination (double consonants)
* Morphological word formation
* Sanskrit-origin words
* Borrowed English words
* Rich suffix combinations
* Context-dependent pronunciation

A single encoder often learns only one representation of text. HKL-VITS solves this by creating a hybrid pipeline where every module handles a specific linguistic or acoustic responsibility.

The final architecture:

```
Kannada Text
       |
       ↓
Text Normalization Layer
       |
       ↓
Kannada G2P Engine
       |
       ↓
Dual Linguistic Representation
       |
       ├───────────────┐
       ↓               ↓
Grapheme Encoder   Phoneme Encoder
       ↓               ↓
       └────── Fusion Attention ──────┘
                     |
                     ↓
          Kannada Linguistic Features
                     |
                     ↓
          Prosody Prediction Network
                     |
                     ↓
          Fine-tuned MMS-VITS Backbone
                     |
                     ↓
              Audio Waveform
```

The key idea:

**Do not ask one model to solve spelling, pronunciation, emotion, rhythm, and audio generation together.**

Instead:

* Rules handle deterministic language processing.
* Neural encoders understand language patterns.
* Prosody models control speaking style.
* VITS generates realistic speech.

---

# 2. Why Kannada Needs a Hybrid TTS System

## 2.1 Kannada is an Abugida Script

Kannada is not a simple alphabet language.

English:

```
C + A + T
=
cat
```

Each letter approximately maps to sound.

Kannada:

```
ಕ
ಕಾ
ಕಿ
ಕೀ
ಕು
ಕೂ
ಕೃ
```

The same base consonant changes pronunciation depending on attached vowel markers.

Example:

```
ಕ
ka

ಕಾ
kaa

ಕಿ
ki

ಕೀ
kii
```

A character-only model sees similar symbols but may not understand phonetic differences.

HKL-VITS introduces explicit phoneme representation.

---

# 3. HKL-VITS Main Components

The system contains multiple integrated models:

## Module Overview

| Module            | Technology                 | Purpose                            |
| ----------------- | -------------------------- | ---------------------------------- |
| Text Normalizer   | Rule based Python          | Cleans and expands input           |
| Kannada G2P       | Linguistic rules + mapping | Converts script into pronunciation |
| Grapheme Encoder  | Transformer                | Understands written Kannada        |
| Phoneme Encoder   | BiLSTM                     | Understands pronunciation sequence |
| Fusion Layer      | Attention mechanism        | Combines both representations      |
| Prosody Model     | BiLSTM network             | Controls rhythm and expression     |
| Emotion Embedding | Neural embedding           | Controls speaking style            |
| MMS-VITS          | Fine-tuned VITS            | Generates speech waveform          |
| Post Processing   | Audio processing           | Improves final output              |

Each component solves a different problem.

---

# 4. Text Normalization Layer

## Purpose

Humans write:

```
ನಾನು 123 ರೂಪಾಯಿ ಕೊಟ್ಟೆ
```

A speech model cannot directly pronounce numbers.

The normalizer converts:

```
123
```

into:

```
ಒಂದು ಎರಡು ಮೂರು
```

Final:

```
ನಾನು ಒಂದು ಎರಡು ಮೂರು ರೂಪಾಯಿ ಕೊಟ್ಟೆ
```

---

## Responsibilities

### Number expansion

Input:

```
50%
```

Output:

```
ಐದು ಸೊನ್ನೆ ಶೇಕಡಾ
```

---

### Symbol expansion

Input:

```
A+B=C
```

Output:

```
A ಪ್ಲಸ್ B ಸಮಾನ C
```

---

### Punctuation preservation

Input:

```
ನೀವು ಹೇಗಿದ್ದೀರಿ?
```

The question mark gives information to the prosody model.

---

Why rule-based?

Because these operations are deterministic.

A neural model may generate:

```
50 → ಐವತ್ತು
```

or

```
ಐದು ಸೊನ್ನೆ
```

depending on training.

Rules guarantee consistency.

---

# 5. Kannada G2P (Grapheme To Phoneme)

## Purpose

G2P converts written Kannada into pronunciation units.

Example:

Input:

```
ಕನ್ನಡ
```

Output:

```
ka nː na ɖa
```

Input:

```
ಕ್ಷ
```

Output:

```
ksha
```

---

## Why G2P is important

The spelling:

```
ಕನ್ನಡ
```

contains:

```
ಕ್ + ಅ + ಣ್ + ನ + ಡ + ಅ
```

The actual sound is different from direct character reading.

A grapheme model sees:

```
ಕ ಣ್ ನ ಡ
```

A phoneme model sees:

```
ka nː na ɖa
```

The second representation directly describes speech.

---

# 6. Grapheme Encoder

## Technology

Transformer Encoder

---

## Purpose

The grapheme encoder understands written Kannada structure.

It learns:

* character relationships
* spelling patterns
* word structure
* context

Example:

Sentence:

```
ಅವನು ಮನೆಗೆ ಹೋದನು
```

The word:

```
ಮನೆಗೆ
```

contains:

```
ಮನೆ + ಗೆ
```

The grapheme encoder understands the written structure.

---

## Why Transformer?

Transformer uses self-attention.

Instead of reading:

```
ಕನ್ನಡ
 ↓
ಕ
 ↓
ನ
 ↓
ನ
```

individually,

it learns relationships:

```
ಕ ↔ ನ ↔ ಡ
```

The model understands the complete word.

---

# 7. Phoneme Encoder (BiLSTM)

## Technology

Bidirectional Long Short-Term Memory Network

---

## Why BiLSTM?

Speech is sequential.

The pronunciation of one sound depends on nearby sounds.

Example:

```
ಅಣ್ಣ
```

Pronunciation:

```
a ṇː ṇa
```

The long consonant:

```
ṇː
```

depends on:

* previous vowel
* next consonant
* word structure

A normal neural network processes:

```
a → ṇ → ṇa
```

only forward.

A BiLSTM processes both directions:

Forward:

```
a → ṇː → ṇa
```

Backward:

```
ṇa → ṇː → a
```

Therefore each phoneme gets information from:

* previous sounds
* future sounds

---

## BiLSTM Internal Working

A normal RNN forgets old information.

Example:

Long word:

```
ಮನೆಗಳಿಂದ
```

contains:

```
ಮನೆ + ಗಳು + ಇಂದ
```

A simple RNN may lose the beginning.

LSTM solves this using memory gates.

---

## LSTM Gates

### Forget Gate

Decides what information to remove.

Example:

```
old pronunciation information
```

---

### Input Gate

Decides what new information to store.

Example:

```
new suffix information
```

---

### Output Gate

Decides what information goes forward.

---

BiLSTM combines:

```
Forward LSTM
+
Backward LSTM
```

Output:

```
complete phoneme context
```

---

## Why use BiLSTM in HKL-VITS?

Because Kannada has:

* suffix changes
* consonant clusters
* pronunciation changes based on surrounding letters

Example:

```
ಮನೆ
ma ne

ಮನೆಗಳಿಂದ
ma ne ga lin da
```

The suffix changes pronunciation behavior.

BiLSTM captures this sequence relationship.

---

# 8. Grapheme + Phoneme Dual Encoder Advantage

Traditional:

```
Text
 |
Encoder
 |
Speech
```

HKL:

```
              Text
                |
       --------------------
       |                  |
  Grapheme             Phoneme
  Encoder              Encoder

Written form       Pronunciation form

       --------------------
                |
             Fusion
```

Advantages:

| Problem       | Grapheme Only | Phoneme Only | HKL       |
| ------------- | ------------- | ------------ | --------- |
| Spelling      | Good          | Weak         | Good      |
| Pronunciation | Medium        | Excellent    | Excellent |
| Conjuncts     | Weak          | Good         | Good      |
| Morphology    | Weak          | Medium       | Strong    |
| OOV words     | Weak          | Better       | Better    |

---

# 9. Fusion Attention Layer

The model now has:

Grapheme features:

```
[batch, sequence, 256]
```

Phoneme features:

```
[batch, sequence, 256]
```

Fusion decides:

"At this position, should I trust spelling or pronunciation more?"

Example:

Normal word:

```
ಮನೆ
```

Mostly phoneme.

Technical word:

```
ಟೆಕ್ನಾಲಜಿ
```

Needs grapheme support.

Attention learns this automatically.

---

# 10. Result of Part 1

HKL-VITS is not a single TTS model.

It is a coordinated system:

```
Rules
 |
Language knowledge
 |
G2P
 |
Dual linguistic understanding
 |
Prosody control
 |
Fine-tuned MMS-VITS speech generation
```

The reason for combining these models:

* Rules provide accuracy
* G2P provides pronunciation
* Transformer provides language understanding
* BiLSTM provides sequential phonology
* Attention provides adaptive fusion
* VITS provides realistic speech generation

**End of Part 1**

I will continue with **Part 2: Prosody Model, Emotion Control, Fine-tuned MMS-VITS integration, training pipeline, parameters, losses, and inference flow.**
