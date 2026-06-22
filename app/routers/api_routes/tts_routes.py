from io import BytesIO

import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
router = APIRouter()


@router.post("/tts")
async def generate_tts(request: Request, data: dict):

    text = data.get("text")

    if not text:

        raise HTTPException(status_code=400, detail="Text required")

    hkl_inference = request.app.state.hkl_inference

    wav = hkl_inference.synthesize(text)

    audio_buffer = BytesIO()

    sf.write(audio_buffer, wav, hkl_inference.config.sample_rate, format="WAV")

    audio_buffer.seek(0)

    return StreamingResponse(audio_buffer, media_type="audio/wav")


@router.get("/model-summary")
async def model_summary(request: Request):

    hkl_inference = request.app.state.hkl_inference

    config = hkl_inference.config

    model = hkl_inference.model

    mms = getattr(model, "mms", None)

    summary = {
        # -----------------------
        # Runtime
        # -----------------------
        "status": "Loaded",
        "device": hkl_inference.device,
        # -----------------------
        # Model
        # -----------------------
        "model": {
            "class": model.__class__.__name__,
            "base_model": config.mms_model_name,
            "checkpoint": config.checkpoint_path,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        # -----------------------
        # MMS Details
        # -----------------------
        "mms": {
            "loaded": mms is not None,
            "class": (mms.__class__.__name__ if mms else None),
        },
        # -----------------------
        # Audio
        # -----------------------
        "audio": {"sample_rate": config.sample_rate, "format": config.audio_format},
        # -----------------------
        # Tokenizer
        # -----------------------
        "tokenizer": {
            "class": hkl_inference.tokenizer.__class__.__name__,
            "model": config.mms_model_name,
        },
        # -----------------------
        # Normalization
        # -----------------------
        "text_normalization": {
            "enabled": config.normalize_text,
            "numbers": config.expand_numbers,
            "symbols": config.expand_symbols,
            "punctuation": config.preserve_punctuation,
        },
        # -----------------------
        # Architecture
        # -----------------------
        "pipeline": [
            "Kannada Text Normalizer",
            "MMS Kannada Tokenizer",
            "Facebook MMS VITS Backbone",
            "HKLTrainerModel Wrapper",
            "VITS Waveform Decoder",
        ],
    }

    return summary
