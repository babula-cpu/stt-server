from __future__ import annotations

import socket
import struct
import threading
import time
from pathlib import Path

import pytest
import uvicorn

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "references"


def load_wav_as_pcm16le_16khz(wav_path: Path) -> bytes:
    """Read a wav file (any format) and convert to PCM16LE mono 16kHz.

    Handles the reference wav which is float32/24kHz.
    """
    with open(wav_path, "rb") as f:
        # Parse RIFF header
        f.seek(12)
        src_rate = 0
        src_channels = 0
        src_bits = 0
        fmt_tag = 0
        data_bytes = b""
        file_size = wav_path.stat().st_size

        while f.tell() < file_size:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            raw = f.read(4)
            if len(raw) < 4:
                break
            chunk_size = struct.unpack("<I", raw)[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                fmt_tag = struct.unpack_from("<H", fmt_data, 0)[0]
                src_channels = struct.unpack_from("<H", fmt_data, 2)[0]
                src_rate = struct.unpack_from("<I", fmt_data, 4)[0]
                src_bits = struct.unpack_from("<H", fmt_data, 14)[0]
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
                break
            else:
                f.seek(chunk_size, 1)

    # Decode samples to float list
    if fmt_tag == 3:  # IEEE float
        n_samples = len(data_bytes) // (src_bits // 8)
        fmt_char = "f" if src_bits == 32 else "d"
        samples = list(struct.unpack(f"<{n_samples}{fmt_char}", data_bytes))
    elif fmt_tag == 1:  # PCM
        if src_bits == 16:
            n_samples = len(data_bytes) // 2
            samples = [s / 32768.0 for s in struct.unpack(f"<{n_samples}h", data_bytes)]
        elif src_bits == 32:
            n_samples = len(data_bytes) // 4
            samples = [s / 2147483648.0 for s in struct.unpack(f"<{n_samples}i", data_bytes)]
        else:
            raise ValueError(f"Unsupported PCM bits: {src_bits}")
    else:
        raise ValueError(f"Unsupported wav format tag: {fmt_tag}")

    # Mono mixdown if needed
    if src_channels > 1:
        mono = []
        for i in range(0, len(samples), src_channels):
            mono.append(sum(samples[i : i + src_channels]) / src_channels)
        samples = mono

    # Resample to 16kHz via linear interpolation
    target_rate = 16000
    if src_rate != target_rate:
        ratio = src_rate / target_rate
        n_target = int(len(samples) / ratio)
        resampled = []
        for i in range(n_target):
            src_pos = i * ratio
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(samples):
                val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            else:
                val = samples[idx]
            resampled.append(val)
        samples = resampled

    # Convert to int16 PCM
    pcm_samples = []
    for s in samples:
        val = max(-1.0, min(1.0, s))
        pcm_samples.append(int(val * 32767))

    return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)


@pytest.fixture(scope="session")
def pcm_audio() -> bytes:
    wav_path = REFERENCES_DIR / "zero_shot_prompt.wav"
    return load_wav_as_pcm16le_16khz(wav_path)


@pytest.fixture(scope="session")
def expected_text() -> str:
    txt_path = REFERENCES_DIR / "zero_shot_prompt.txt"
    return txt_path.read_text(encoding="utf-8").strip()


@pytest.fixture(scope="session")
def server_url():
    """Start a real uvicorn server in a background thread, yield its URL."""
    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(
        "app:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start in time")

    yield f"ws://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


# ── Integration test fixtures ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def references_dir() -> Path:
    """Path to reference audio and transcript files."""
    return REFERENCES_DIR


@pytest.fixture(scope="session")
def reference_audio_files(references_dir: Path) -> list[Path]:
    """Discover all .wav files in references directory."""
    return sorted(references_dir.glob("*.wav"))


def find_transcript_for_audio(audio_path: Path, references_dir: Path) -> Path | None:
    """Find matching .txt file for audio file.

    Pairing rule: audio.wav -> audio.txt
    """
    txt_path = references_dir / f"{audio_path.stem}.txt"
    return txt_path if txt_path.exists() else None
