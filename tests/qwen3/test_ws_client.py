#!/usr/bin/env python3
"""Simple WebSocket test client."""

import os
import sys
from pathlib import Path

# Skip if running with mock (needs running server)
if os.environ.get("STT_BACKEND") == "mock":
    import pytest
    pytest.skip("Requires running server (STT_BACKEND=qwen3)", allow_module_level=True)

import asyncio
import json
import websockets

async def test():
    url = "ws://localhost:8000/v1/stt/ws?call_id=test123&sample_rate=16000"
    async with websockets.connect(url, open_timeout=30) as ws:
        print("Connected, waiting for ready...")
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
        print(f"Received: {msg}")

        if msg["type"] != "ready":
            print(f"Expected ready, got {msg['type']}")
            return

        # Read audio file
        from tests.conftest import load_wav_as_pcm16le_16khz

        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        wav_path = project_root / "references" / "zero_shot_prompt.wav"
        pcm_audio = load_wav_as_pcm16le_16khz(wav_path)
        print(f"Sending {len(pcm_audio)} bytes of audio...")

        # Send audio in chunks
        chunk_size = 3200
        for i in range(0, len(pcm_audio), chunk_size):
            chunk = pcm_audio[i:i+chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.01)  # small delay

        # Send commit to trigger finalization
        print("Sending commit...")
        await ws.send(json.dumps({"type": "commit"}))

        print("Waiting for final...")
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                msg = json.loads(raw)
                print(f"Received: {msg}")
                if msg["type"] == "final":
                    print(f"Final transcript: {msg['text']}")
                    break
                elif msg["type"] == "error":
                    print(f"Error: {msg}")
                    break
        except asyncio.TimeoutError:
            print("Timeout waiting for final")

if __name__ == "__main__":
    asyncio.run(test())
