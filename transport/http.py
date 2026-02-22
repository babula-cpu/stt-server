"""HTTP endpoints for offline transcription."""

import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backends.registry import create_backend
from config import settings
from protocol.audio import WavDecodeError, decode_wav_to_pcm16le

router = APIRouter()


@router.post("/v1/stt/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "auto",
) -> JSONResponse:
    """Transcribe audio file offline (non-streaming).

    Args:
        file: WAV audio file
        language: Language code (e.g., 'zh', 'en', 'auto')

    Returns:
        JSON with transcription text, duration_ms, and latency_ms

    Raises:
        HTTPException: 413 if file too large, 422 if invalid WAV
    """
    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max: {settings.max_upload_bytes}",
        )

    # Decode WAV
    try:
        pcm_data = decode_wav_to_pcm16le(content)
    except WavDecodeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Calculate duration_ms
    duration_ms = len(pcm_data) / 2 / 16000 * 1000

    # Run inference in thread
    start_time = time.perf_counter()

    def run_inference():
        backend = create_backend(settings.backend)
        backend.configure(16000, language)
        backend.push_audio(pcm_data)
        result = backend.finalize()
        backend.close()
        return result

    # Execute in thread pool
    import asyncio

    result = await asyncio.to_thread(run_inference)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return JSONResponse(
        content={
            "text": result.text,
            "duration_ms": duration_ms,
            "latency_ms": latency_ms,
        }
    )
