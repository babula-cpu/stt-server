from __future__ import annotations

import os
import struct
import time
from pathlib import Path

from backends.base import ASRResult

_DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "references"

# Env-configurable simulated inference delay (ms)
_MOCK_INFER_MS = float(os.environ.get("MOCK_INFER_MS", "5"))


def _parse_wav_duration(wav_path: Path) -> float:
    """Parse wav header to extract duration in seconds. Handles PCM and float formats."""
    with open(wav_path, "rb") as f:
        # Skip RIFF header (12 bytes)
        f.seek(12)
        sample_rate = 0
        channels = 0
        bits_per_sample = 0
        data_size = 0
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
                channels = struct.unpack_from("<H", fmt_data, 2)[0]
                sample_rate = struct.unpack_from("<I", fmt_data, 4)[0]
                bits_per_sample = struct.unpack_from("<H", fmt_data, 14)[0]
            elif chunk_id == b"data":
                data_size = chunk_size
                break
            else:
                f.seek(chunk_size, 1)

        if sample_rate == 0 or channels == 0 or bits_per_sample == 0:
            raise ValueError(f"Could not parse wav header: {wav_path}")

        bytes_per_sample = bits_per_sample // 8
        total_samples = data_size // (bytes_per_sample * channels)
        return total_samples / sample_rate


class MockBackend:
    """Fixture-based replay backend for testing without a GPU."""

    def __init__(self, fixture_dir: str | Path | None = None) -> None:
        fixture_path = Path(fixture_dir) if fixture_dir else _DEFAULT_FIXTURE_DIR
        txt_path = fixture_path / "zero_shot_prompt.txt"
        wav_path = fixture_path / "zero_shot_prompt.wav"

        self._target_text = txt_path.read_text(encoding="utf-8").strip()
        self._wav_duration = _parse_wav_duration(wav_path)

        self._sample_rate: int = 16000
        self._buf_bytes: int = 0
        self._total_expected_bytes: int = 0
        self._endpoint_fired: bool = False
        self._configured: bool = False

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        # Expected PCM16LE bytes = duration * sample_rate * 2 (16-bit mono)
        self._total_expected_bytes = int(self._wav_duration * sample_rate * 2)
        self._buf_bytes = 0
        self._endpoint_fired = False
        self._configured = True

    def push_audio(self, pcm_data: bytes) -> None:
        self._buf_bytes += len(pcm_data)

    def get_partial(self) -> ASRResult | None:
        if self._buf_bytes == 0 or self._total_expected_bytes == 0:
            return None

        ratio = min(self._buf_bytes / self._total_expected_bytes, 1.0)
        char_count = max(1, int(len(self._target_text) * ratio))
        text = self._target_text[:char_count]

        if _MOCK_INFER_MS > 0:
            time.sleep(_MOCK_INFER_MS / 1000)

        return ASRResult(text=text, is_partial=True, is_endpoint=False)

    def finalize(self) -> ASRResult:
        if _MOCK_INFER_MS > 0:
            time.sleep(_MOCK_INFER_MS / 1000)
        result = ASRResult(text=self._target_text, is_partial=False, is_endpoint=True)
        self._buf_bytes = 0
        return result

    def detect_endpoint(self) -> bool:
        if self._endpoint_fired:
            return False
        if self._total_expected_bytes > 0 and self._buf_bytes >= self._total_expected_bytes * 0.95:
            self._endpoint_fired = True
            return True
        return False

    def reset_segment(self) -> None:
        self._buf_bytes = 0
        self._endpoint_fired = False

    def close(self) -> None:
        pass
