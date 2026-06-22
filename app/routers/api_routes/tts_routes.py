from io import BytesIO

import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
router = APIRouter()



@router.post("/tts")
async def generate_tts(
    request: Request,
    data: dict
):

    text = data.get("text")


    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text required"
        )



    hkl_inference = request.app.state.hkl_inference



    wav = hkl_inference.synthesize(text)



    audio_buffer = BytesIO()



    sf.write(
        audio_buffer,
        wav,
        16000,
        format="WAV"
    )



    audio_buffer.seek(0)



    return StreamingResponse(

        audio_buffer,

        media_type="audio/wav"

    )






@router.get("/model-summary")
async def model_summary(
    request: Request
):


    hkl_inference = request.app.state.hkl_inference



    model = getattr(
        hkl_inference,
        "model",
        None
    )



    summary = {


        "name":
        "HKL-VITS Hybrid Kannada TTS",


        "status":
        "Loaded",


        "sample_rate":
        16000,


        "description":
        (
            "Hybrid Kannada Text To Speech system "
            "combining MMS-VITS backbone with "
            "Kannada linguistic encoder, phoneme "
            "processing and prosody modeling."
        ),


        "architecture":[

            "Facebook MMS Kannada VITS Backbone",

            "Kannada Grapheme Encoder",

            "Kannada Phoneme Encoder",

            "Fusion Attention Layer",

            "Kannada Prosody Predictor"

        ]

    }





    if model:


        summary["model_class"] = (
            model.__class__.__name__
        )


        summary["parameters"] = (

            sum(
                p.numel()
                for p in model.parameters()
            )

        )



    else:


        summary["model_class"] = (

            hkl_inference
            .__class__
            .__name__

        )



    return summary