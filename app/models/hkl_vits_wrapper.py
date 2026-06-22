"""
HKL-VITS Hybrid Kannada TTS

Full Hybrid Model Wrapper

MMS-TTS Kannada Backbone
+
HKL Linguistic Encoder
+
Prosody Predictor
app/models/hkl_vits_wrapper.py
"""

import torch.nn as nn

from transformers import VitsModel


class HKLVITS(nn.Module):

    def __init__(
        self, mms_model_name, linguistic_encoder, prosody_model, hkl_hidden_dim=256
    ):

        super().__init__()

        # ======================================
        # MMS Backbone
        # ======================================

        self.mms = VitsModel.from_pretrained(mms_model_name)

        self.mms_hidden_dim = self.mms.config.hidden_size

        # ======================================
        # HKL Components
        # ======================================

        self.linguistic_encoder = linguistic_encoder

        self.prosody_model = prosody_model

        # ======================================
        # HKL -> MMS Adapter
        # ======================================

        self.mms_projection = nn.Linear(hkl_hidden_dim, self.mms_hidden_dim)

    def forward(self, input_ids, phoneme_ids):

        # --------------------------------------
        # HKL linguistic representation
        # --------------------------------------

        hkl_features = self.linguistic_encoder(input_ids, phoneme_ids)

        # --------------------------------------
        # Prosody prediction
        # --------------------------------------

        prosody = self.prosody_model(hkl_features)

        # --------------------------------------
        # Adapt to MMS dimension
        # --------------------------------------

        mms_features = self.mms_projection(hkl_features)

        # --------------------------------------
        # MMS synthesis
        # --------------------------------------

        mms_output = self.mms(input_ids=input_ids)

        return {
            "waveform": mms_output.waveform,
            "spectrogram": mms_output.spectrogram,
            "hkl_features": hkl_features,
            "mms_features": mms_features,
            "prosody": prosody,
        }
