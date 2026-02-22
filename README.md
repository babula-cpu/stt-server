# realtime-stt-ws

Real-time speech-to-text WebSocket server with pluggable ASR backends.

## Architecture

```
WebSocket Client
      │
      ▼
┌─────────────────────────────────────┐
│  FastAPI (async)                    │
│                                     │
│  _recv_pump ──► in_queue            │
│                    │                │
│              run_worker (thread)    │
│                    │                │
│  _send_pump ◄── out_queue           │
└─────────────────────────────────────┘
      │
      ▼
  ASR Backend (Qwen3 / Mock)
```

- **Async I/O** for WebSocket recv/send via FastAPI
- **Sync worker thread** for inference (bridged with janus queues)
- **Pluggable backends** via `ASRBackend` protocol
- **Energy-based VAD** for utterance endpoint detection
- **Prometheus metrics** at `/metrics` and health check at `/health`

## Backends

| Backend | Description | Requires |
|---------|-------------|----------|
| `qwen3` | Qwen3-ASR-1.7B via embedded vLLM (in-process GPU) | CUDA + `vllm[audio]` |
| `mock` | Fixture-based replay for testing | Nothing extra |

Set via `STT_BACKEND` environment variable (default: `qwen3`).

### Backend Roadmap

| Phase | Backend | Model | Status |
|-------|---------|-------|--------|
| Phase 1 | `qwen3` | Qwen3-ASR-1.7B (vLLM) | Done |
| Phase 2 | `whisper` | Whisper | Planned |
| Phase 3 | `nemo` | NVIDIA NeMo | Planned |

## Setup

```bash
# Install core + dev dependencies
uv sync --group dev

# With Qwen3 backend (requires CUDA)
uv sync --group dev --extra qwen3
```

## Usage

```bash
# Run with mock backend (no GPU needed)
STT_BACKEND=mock uvicorn app:app --host 0.0.0.0 --port 8000

# Run with Qwen3 backend (requires CUDA)
STT_BACKEND=qwen3 uvicorn app:app --host 0.0.0.0 --port 8000
```

### WebSocket Protocol

Connect to `ws://host:port/v1/stt/ws` with query parameters:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `call_id` | yes | - | Unique call identifier |
| `sample_rate` | yes | - | Must be `16000` |
| `codec` | no | `pcm_s16le` | Audio codec |
| `channels` | no | `1` | Must be mono |
| `frame_ms` | no | `20` | Frame duration in ms |
| `partial_ms` | no | `500` | Partial result interval in ms |
| `language` | no | `auto` | BCP-47 code (e.g. `zh`, `en`, `ja`) |
| `hotwords` | no | - | Comma-separated hotwords |

**Send:** binary PCM16LE audio frames, or JSON `{"type": "commit"}` to force-finalize.

**Receive:** JSON messages:

### HTTP Interface (Offline)

For batch transcription, use `POST /v1/stt/transcribe`:

```bash
# Upload a WAV file for transcription
curl -X POST http://localhost:8000/v1/stt/transcribe \
  -F "file=@audio.wav" \
  -G -d "language=zh"

# Response:
# {
#   "text": "转写文本内容",
#   "duration_ms": 3200.0,
#   "latency_ms": 450.3
# }
```

Supported audio formats: WAV (PCM 16/32-bit, IEEE Float 32/64-bit). Audio is automatically converted to 16kHz mono.

```jsonc
// Connection ready
{"type": "ready", "call_id": "...", "sample_rate": 16000, ...}

// Partial transcription
{"type": "partial", "call_id": "...", "segment_seq": 0, "text": "hello", "final": false}

// Final transcription (after endpoint or commit)
{"type": "final", "call_id": "...", "segment_seq": 0, "text": "hello world", "final": true,
 "first_token_ms": 120, "latency_ms": 250, "segment_duration_ms": 3200}

// Error
{"type": "error", "call_id": "...", "code": 1008, "message": "..."}
```

## Configuration

All settings use the `STT_` env prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_BACKEND` | `qwen3` | Backend name (`qwen3` or `mock`) |
| `STT_HOST` | `0.0.0.0` | Server bind host |
| `STT_PORT` | `8000` | Server bind port |
| `STT_IN_QUEUE_SIZE` | `200` | Input queue capacity |
| `STT_OUT_QUEUE_SIZE` | `50` | Output queue capacity |
| `STT_MAX_UPLOAD_BYTES` | `104857600` | Max upload size (100MB) |
| `STT_VLLM_MODEL` | `Qwen/Qwen3-ASR-1.7B` | vLLM model name |
| `STT_VLLM_GPU_MEMORY` | `0.8` | GPU memory utilization (0-1) |
| `STT_VLLM_MAX_TOKENS` | `512` | Max output tokens |
| `STT_VAD_THRESHOLD` | `300` | RMS energy threshold for speech |
| `STT_VAD_SILENCE_MS` | `700` | Silence duration to trigger endpoint |

## Testing

```bash
# Run all tests (uses mock backend)
STT_BACKEND=mock pytest tests/ -v
```

## Project Structure

```
├── app.py                  # FastAPI entry point
├── config.py               # Pydantic settings
├── backends/
│   ├── base.py             # ASRBackend protocol + ASRResult
│   ├── registry.py         # Backend factory
│   ├── mock.py             # Fixture-based mock backend
│   └── qwen3.py            # vLLM Qwen3 backend
├── inference/
│   └── worker.py           # Sync inference worker thread
├── protocol/
│   └── schema.py           # Message schemas & validation
├── session/
│   └── state.py            # Per-connection session state
├── transport/
│   └── ws.py               # WebSocket router & pumps
├── observability/
│   ├── logging.py          # Structured logging (structlog)
│   └── metrics.py          # Prometheus metrics
├── references/             # Test fixtures (audio + text)
└── tests/                  # Unit & integration tests
```
