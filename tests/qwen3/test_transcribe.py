"""HTTP integration tests for /v1/stt/transcribe endpoint."""

import io
import os
from pathlib import Path

import httpx
import pytest

# Reference audio file
REFERENCES_DIR = Path(__file__).resolve().parent.parent.parent / "references"
ZERO_SHOT_WAV = REFERENCES_DIR / "zero_shot_prompt.wav"
ZERO_SHOT_TEXT = REFERENCES_DIR / "zero_shot_prompt.txt"


def normalize_text(text: str) -> str:
    """Normalize text for comparison: strip + collapse whitespace + remove trailing punctuation."""
    import re
    return re.sub(r"\s+", " ", text.strip()).rstrip("。.")


def test_transcribe_success():
    """Test successful transcription of WAV file."""
    with open(ZERO_SHOT_WAV, "rb") as f:
        files = {"file": ("zero_shot_prompt.wav", f, "audio/wav")}
        response = httpx.post(
            "http://localhost:8000/v1/stt/transcribe",
            files=files,
            timeout=60.0,
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "duration_ms" in data
    assert "latency_ms" in data

    # Verify transcription matches reference
    reference = ZERO_SHOT_TEXT.read_text(encoding="utf-8").strip()
    assert normalize_text(data["text"]) == normalize_text(reference)


def test_transcribe_zh_language():
    """Test transcription with explicit Chinese language."""
    with open(ZERO_SHOT_WAV, "rb") as f:
        files = {"file": ("zero_shot_prompt.wav", f, "audio/wav")}
        response = httpx.post(
            "http://localhost:8000/v1/stt/transcribe?language=zh",
            files=files,
            timeout=60.0,
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert len(data["text"]) > 0


def test_transcribe_en_language():
    """Test transcription with explicit English language."""
    # Note: The reference audio is Chinese, so result may not be accurate
    # This test verifies the language parameter is accepted
    with open(ZERO_SHOT_WAV, "rb") as f:
        files = {"file": ("zero_shot_prompt.wav", f, "audio/wav")}
        response = httpx.post(
            "http://localhost:8000/v1/stt/transcribe?language=en",
            files=files,
            timeout=60.0,
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data


def test_transcribe_response_format():
    """Test response contains all required fields."""
    with open(ZERO_SHOT_WAV, "rb") as f:
        files = {"file": ("zero_shot_prompt.wav", f, "audio/wav")}
        response = httpx.post(
            "http://localhost:8000/v1/stt/transcribe",
            files=files,
            timeout=60.0,
        )

    assert response.status_code == 200
    data = response.json()

    # Check field types
    assert isinstance(data["text"], str)
    assert isinstance(data["duration_ms"], (int, float))
    assert isinstance(data["latency_ms"], (int, float))

    # Check reasonable values
    assert data["duration_ms"] > 0
    assert data["latency_ms"] > 0
    assert len(data["text"]) > 0


def test_transcribe_invalid_file():
    """Test non-WAV file returns 422."""
    # Create a text file pretending to be audio
    fake_audio = io.BytesIO(b"not a wav file content")

    files = {"file": ("test.txt", fake_audio, "text/plain")}
    response = httpx.post(
        "http://localhost:8000/v1/stt/transcribe",
        files=files,
        timeout=10.0,
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_transcribe_empty_file():
    """Test empty file returns 422."""
    empty_file = io.BytesIO(b"")

    files = {"file": ("empty.wav", empty_file, "audio/wav")}
    response = httpx.post(
        "http://localhost:8000/v1/stt/transcribe",
        files=files,
        timeout=10.0,
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_transcribe_file_too_large():
    """Test file exceeding size limit returns 413."""
    # Read existing file to get format
    with open(ZERO_SHOT_WAV, "rb") as f:
        wav_data = f.read()

    # Create large fake file by repeating
    large_data = wav_data * 1000  # Make it large

    files = {"file": ("large.wav", io.BytesIO(large_data), "audio/wav")}
    response = httpx.post(
        "http://localhost:8000/v1/stt/transcribe",
        files=files,
        timeout=10.0,
    )

    # Should return 413 if limit is enforced, or 422 if it passes that check first
    assert response.status_code in [413, 422]
    data = response.json()
    assert "detail" in data
