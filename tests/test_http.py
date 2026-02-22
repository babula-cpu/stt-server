from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import app

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "references"


client = TestClient(app)


def test_transcribe_wav():
    wav_path = REFERENCES_DIR / "zero_shot_prompt.wav"
    txt_path = REFERENCES_DIR / "zero_shot_prompt.txt"
    expected = txt_path.read_text(encoding="utf-8").strip()

    with open(wav_path, "rb") as f:
        resp = client.post(
            "/v1/stt/transcribe",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"language": "auto"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == expected
    assert body["duration_ms"] > 0
    assert body["latency_ms"] > 0


def test_transcribe_invalid_format():
    resp = client.post(
        "/v1/stt/transcribe",
        files={"file": ("test.mp3", b"not a wav file", "audio/mpeg")},
    )
    assert resp.status_code == 422
    assert "WAV" in resp.json()["detail"]


def test_transcribe_missing_file():
    resp = client.post("/v1/stt/transcribe")
    assert resp.status_code == 422
