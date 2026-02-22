"""Real audio integration tests for Qwen3-ASR via vLLM."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import websockets

from tests.conftest import load_wav_as_pcm16le_16khz

# Skip if not using qwen3 backend (requires real GPU inference)
skip_if_mock = pytest.mark.skipif(
    os.environ.get("STT_BACKEND") != "qwen3",
    reason="Real audio tests require STT_BACKEND=qwen3"
)


@dataclass
class TranscriptionResult:
    """Result of a transcription test."""
    audio_file: Path
    transcript_file: Path
    transcript: str
    reference: str
    latency_ms: float
    passed: bool
    error: str | None = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: strip + collapse whitespace + remove trailing punctuation."""
    return re.sub(r"\s+", " ", text.strip()).rstrip("。.")


async def transcribe_via_websocket(
    server_url: str, pcm_audio: bytes, call_id: str, timeout: float = 60.0
) -> tuple[str, float]:
    """Send audio via WebSocket and return transcript + latency.

    Args:
        server_url: WebSocket server URL
        pcm_audio: Raw PCM audio data
        call_id: Unique call identifier
        timeout: Connection timeout in seconds (default 60s for first load)

    Returns:
        Tuple of (transcript, latency_ms)
    """
    url = f"{server_url}/v1/stt/ws?call_id={call_id}&sample_rate=16000"

    async with websockets.connect(url, open_timeout=timeout) as ws:
        # Wait for ready (may take time on first load)
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg["type"] != "ready":
            raise RuntimeError(f"Expected ready, got {msg['type']}")

        # Send audio in chunks (100ms = 3200 bytes at 16kHz)
        chunk_size = 3200
        for i in range(0, len(pcm_audio), chunk_size):
            chunk = pcm_audio[i : i + chunk_size]
            await ws.send(chunk)

        # Send commit message to finalize transcription
        await ws.send(json.dumps({"type": "commit"}))

        # Collect final message
        start_time = time.perf_counter()
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            if msg["type"] == "final":
                latency_ms = (time.perf_counter() - start_time) * 1000
                return msg["text"], latency_ms
            elif msg["type"] == "error":
                raise RuntimeError(f"Transcription error: {msg.get('message', 'unknown')}")


def find_transcript_for_audio(audio_path: Path, references_dir: Path) -> Path | None:
    """Find matching .txt file for audio file.

    Pairing rule: audio.wav -> audio.txt
    """
    txt_path = references_dir / f"{audio_path.stem}.txt"
    return txt_path if txt_path.exists() else None


@skip_if_mock
class TestRealAudioASR:
    """Integration tests using real audio files via WebSocket."""

    def test_real_audio_transcription(
        self, reference_audio_files: list[Path], references_dir: Path
    ):
        """Test all reference audio files against their transcripts.

        Connects to server at ws://localhost:8000 (must be started separately).
        Outputs per-sample report and summary.
        """
        # Connect to already-running server
        server_url = "ws://localhost:8000"
        """Test all reference audio files against their transcripts.

        Outputs per-sample report and summary.
        """
        results: list[TranscriptionResult] = []

        for audio_file in reference_audio_files:
            # Find matching transcript
            transcript_file = find_transcript_for_audio(audio_file, references_dir)
            if transcript_file is None:
                pytest.skip(f"No transcript found for {audio_file.name}")
                continue

            # Load audio
            pcm_audio = load_wav_as_pcm16le_16khz(audio_file)
            reference = transcript_file.read_text(encoding="utf-8").strip()

            # Transcribe via WebSocket
            try:
                transcript, latency_ms = asyncio.run(
                    transcribe_via_websocket(
                        server_url,
                        pcm_audio,
                        f"test_{audio_file.stem}",
                    )
                )
                passed = normalize_text(transcript) == normalize_text(reference)
                results.append(
                    TranscriptionResult(
                        audio_file=audio_file,
                        transcript_file=transcript_file,
                        transcript=transcript,
                        reference=reference,
                        latency_ms=latency_ms,
                        passed=passed,
                    )
                )
            except Exception as e:
                import traceback
                error_detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                results.append(
                    TranscriptionResult(
                        audio_file=audio_file,
                        transcript_file=transcript_file,
                        transcript="",
                        reference=reference,
                        latency_ms=0,
                        passed=False,
                        error=error_detail,
                    )
                )

        # Print report
        print("\n" + "=" * 60)
        print("Real Audio ASR Test Report")
        print("=" * 60)

        for r in results:
            print(f"\nSample: {r.audio_file.name}")
            print(f"  Latency: {r.latency_ms:.0f}ms")
            print(f"  Transcript: {r.transcript}")
            print(f"  Reference: {r.reference}")
            if r.error:
                print(f"  Error: {r.error}")
            print(f"  Status: {'PASS' if r.passed else 'FAIL'}")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        sorted_latencies = sorted(latencies)
        median_latency = (
            sorted_latencies[len(sorted_latencies) // 2] if latencies else 0
        )

        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"  Total: {total}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Avg Latency: {avg_latency:.0f}ms")
        print(f"  Median Latency: {median_latency:.0f}ms")
        print("=" * 60)

        # Assert all passed
        assert failed == 0, f"{failed} test(s) failed - see report above"
