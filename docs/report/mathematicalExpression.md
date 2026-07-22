
# Mathematical Formulation of HKL-VITS

## 1. Input Text Representation

Let the input Kannada sentence be

[
T={t_1,t_2,\ldots,t_n}
]

where

* (t_i) is the (i^{th}) Kannada character
* (n) is the sentence length.

Example

[
T={\text{ನ},\text{ಮ},\text{ಸ},\text{್ಕಾ},\text{ರ}}
]

This serves as the input to the HKL-VITS pipeline.

---

# 2. Text Normalization

The normalization function converts raw text into speech-friendly text.

[
T_N=f_{norm}(T)
]

where

* (T) = original text
* (T_N) = normalized text

Example

[
\text{₹2500}
]

becomes

[
\text{ರೂಪಾಯಿ ಎರಡು ಸಾವಿರದ ಐನೂರು}
]

The normalization function includes

* number expansion
* currency expansion
* symbol expansion
* punctuation handling

---

# 3. Grapheme-to-Phoneme Conversion

The G2P module maps normalized Kannada text into a phoneme sequence.

[
P=f_{g2p}(T_N)
]

where

[
P={p_1,p_2,\ldots,p_m}
]

Example

[
\text{ಕನ್ನಡ}
]

↓

[
{ka,;ṇa,;na,;ḍa}
]

This phoneme representation captures pronunciation explicitly.

---

# 4. Grapheme Embedding

Each Kannada character is converted into a dense vector using an embedding layer.

[
E_g = Embedding(T_N)
]

where

[
E_g\in\mathbb{R}^{n\times d}
]

Here

* (n) = sequence length
* (d=256)

In your implementation,

[
d=256
]

Each character is therefore represented by a 256-dimensional feature vector.

---

# 5. Transformer Grapheme Encoder

The Transformer encoder learns contextual relationships among characters.

[
H_g=Transformer(E_g)
]

where

[
H_g\in\mathbb{R}^{n\times256}
]

Internally, self-attention is computed as

[
Attention(Q,K,V)
================

Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V
]

where

* (Q) = Query
* (K) = Key
* (V) = Value

This allows every Kannada character to attend to every other character in the sentence.

---

# 6. Phoneme Embedding

Similarly,

[
E_p=Embedding(P)
]

where

[
E_p\in\mathbb{R}^{m\times256}
]

---

# 7. BiLSTM Phoneme Encoder

Forward LSTM

[
\overrightarrow{h_t}
====================

LSTM_f
(p_t,\overrightarrow{h_{t-1}})
]

Backward LSTM

[
\overleftarrow{h_t}
===================

LSTM_b
(p_t,\overleftarrow{h_{t+1}})
]

Final phoneme representation

[
H_p
===

[
\overrightarrow{h_t};
\overleftarrow{h_t}
]
]

where

[
H_p
\in
\mathbb{R}^{m\times256}
]

The concatenation combines information from both past and future phoneme contexts.

---

# 8. Fusion Attention

The grapheme and phoneme features are fused using multi-head attention.

[
F
=

Attention(H_g,H_p,H_p)
]

where

[
F
\in
\mathbb{R}^{n\times256}
]

This enables the model to integrate both spelling and pronunciation information.

---

# 9. Prosody Prediction

The fused representation is passed to the prosody model.

[
R
=

BiLSTM(F)
]

The prosody outputs are

[
Pitch
=====

W_pR+b_p
]

[
Duration
========

W_dR+b_d
]

[
Energy
======

W_eR+b_e
]

[
Pause
=====

W_{pa}R+b_{pa}
]

where

* (W) = learnable weight matrix
* (b) = bias vector.

---

# 10. Emotion Embedding (Part 3)

If emotion conditioning is enabled,

[
E_{emotion}
===========

Embedding(c)
]

where

[
c
\in
{
Neutral,
Happy,
Sad,
Angry,
Question
}
]

The linguistic representation becomes

[
F'
==

F+E_{emotion}
]

---

# 11. Adapter Projection

The HKL encoder outputs

[
256
]

dimensions while MMS-VITS expects

[
192
]

dimensions.

A linear projection is applied.

[
Z
=

WF'+b
]

where

[
W
\in
\mathbb{R}^{192\times256}
]

Output

[
Z
\in
\mathbb{R}^{n\times192}
]

---

# 12. Fine-Tuned MMS-VITS

The projected representation is passed into the acoustic model.

[
Y
=

MMS_{VITS}(Z)
]

where

[
Y
=

Generated\ Waveform
]

This represents the synthesized Kannada speech signal.

---

# 13. Overall Loss Function

The total training objective combines multiple losses.

[
L_{total}
=========

L_{acoustic}
+
\lambda_1L_{prosody}
+
\lambda_2L_{phoneme}
]

where

* (L_{acoustic}) = waveform or mel-spectrogram reconstruction loss
* (L_{prosody}) = pitch, duration, energy prediction loss
* (L_{phoneme}) = pronunciation consistency loss
* (\lambda_1,\lambda_2) = weighting coefficients.

---

# 14. End-to-End HKL-VITS Equation

The complete pipeline can be summarized as

[
Y
=

MMS_{VITS}
\left(
Adapter
\left(
Prosody
\left(
Fusion
\left(
Transformer(T_N),
BiLSTM(G2P(T_N))
\right)
\right)
\right)
\right)
]

where

[
T_N
===

Normalization(T)
]

and (Y) is the synthesized Kannada speech waveform.

---

# Complete Mathematical Pipeline

[
T
\rightarrow
T_N
\rightarrow
P
\rightarrow
(E_g,E_p)
\rightarrow
(H_g,H_p)
\rightarrow
F
\rightarrow
R
\rightarrow
F'
\rightarrow
Z
\rightarrow
Y
]

where:

* (T): Input Kannada text
* (T_N): Normalized text
* (P): Phoneme sequence
* (E_g, E_p): Grapheme and phoneme embeddings
* (H_g, H_p): Transformer and BiLSTM encoded features
* (F): Fusion attention output
* (R): Prosody representation
* (F'): Emotion-conditioned representation (if enabled)
* (Z): Adapter output (192-dimensional features)
* (Y): Generated speech waveform

