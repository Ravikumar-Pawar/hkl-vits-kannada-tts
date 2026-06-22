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

    sf.write(audio_buffer, wav, 16000, format="WAV")

    audio_buffer.seek(0)

    return StreamingResponse(audio_buffer, media_type="audio/wav")
