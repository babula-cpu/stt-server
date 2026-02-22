import asyncio
import json
import re

import pytest
import websockets


@pytest.mark.asyncio
async def test_happy_path(server_url: str, pcm_audio: bytes, expected_text: str):
    """Connect, stream wav, receive final, compare text."""
    url = f"{server_url}/v1/stt/ws?call_id=00000000000000007000000000000001&sample_rate=16000"
    async with websockets.connect(url) as ws:
        # Wait for ready
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready"
        assert msg["call_id"] == "00000000000000007000000000000001"

        # Send audio in 100ms chunks (3200 bytes at 16kHz 16-bit mono)
        chunk_size = 3200
        for i in range(0, len(pcm_audio), chunk_size):
            chunk = pcm_audio[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.02)  # pace sending

        # Collect messages until we get a final
        finals = []
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg["type"] == "final":
                    finals.append(msg)
                    break
                elif msg["type"] == "error":
                    pytest.fail(f"Unexpected error: {msg}")
        except asyncio.TimeoutError:
            pytest.fail("Timed out waiting for final message")

        assert len(finals) >= 1
        # Normalize: strip + collapse whitespace
        received = re.sub(r"\s+", " ", finals[0]["text"].strip())
        expected = re.sub(r"\s+", " ", expected_text.strip())
        assert received == expected

        # Verify final has required timing fields
        assert "first_token_ms" in finals[0]
        assert "latency_ms" in finals[0]
        assert "segment_duration_ms" in finals[0]
        assert finals[0]["segment_seq"] == 1


@pytest.mark.asyncio
async def test_invalid_sample_rate(server_url: str):
    """Connecting with sample_rate=8000 must get error(1008) and close."""
    url = f"{server_url}/v1/stt/ws?call_id=00000000000000007000000000000002&sample_rate=8000"
    async with websockets.connect(url) as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        msg = json.loads(raw)
        assert msg["type"] == "error"
        assert msg["code"] == 1008
        assert "16000" in msg["message"]


@pytest.mark.asyncio
async def test_backpressure_overflow(server_url: str):
    """Blast audio frames to trigger input queue overflow."""
    url = f"{server_url}/v1/stt/ws?call_id=00000000000000007000000000000003&sample_rate=16000"
    async with websockets.connect(url) as ws:
        # Wait for ready
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready"

        # Blast frames as fast as possible
        chunk = b"\x00" * 3200
        error_received = False
        try:
            for _ in range(10000):
                await ws.send(chunk)
            # After blasting, read responses
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg["type"] == "error":
                    assert msg["code"] == 1008
                    assert "overflow" in msg["message"].lower()
                    error_received = True
                    break
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
            pass

        assert error_received, "Expected overflow error"


@pytest.mark.asyncio
async def test_commit_flow(server_url: str, pcm_audio: bytes):
    """Send audio, commit, get final(seq=1), send more, commit, get final(seq=2)."""
    url = f"{server_url}/v1/stt/ws?call_id=00000000000000007000000000000004&sample_rate=16000&partial_ms=50"
    async with websockets.connect(url) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready"

        # Send ~0.5s of audio (8000 bytes at 16kHz 16-bit)
        chunk_size = 3200
        bytes_to_send = 8000
        for i in range(0, min(bytes_to_send, len(pcm_audio)), chunk_size):
            await ws.send(pcm_audio[i : i + chunk_size])
            await asyncio.sleep(0.01)

        # Send commit
        await ws.send(json.dumps({"type": "commit"}))

        # Wait for final with seq=1
        final1 = None
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg["type"] == "final":
                    final1 = msg
                    break
                elif msg["type"] == "error":
                    pytest.fail(f"Unexpected error: {msg}")
        except asyncio.TimeoutError:
            pytest.fail("Timed out waiting for first final")

        assert final1 is not None
        assert final1["segment_seq"] == 1

        # Send more audio for segment 2
        for i in range(0, min(bytes_to_send, len(pcm_audio)), chunk_size):
            await ws.send(pcm_audio[i : i + chunk_size])
            await asyncio.sleep(0.01)

        # Commit again
        await ws.send(json.dumps({"type": "commit"}))

        # Wait for final with seq=2
        final2 = None
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg["type"] == "final":
                    final2 = msg
                    break
                elif msg["type"] == "error":
                    pytest.fail(f"Unexpected error: {msg}")
        except asyncio.TimeoutError:
            pytest.fail("Timed out waiting for second final")

        assert final2 is not None
        assert final2["segment_seq"] == 2
