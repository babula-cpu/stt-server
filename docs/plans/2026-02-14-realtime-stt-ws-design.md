# realtime-stt-ws — Implementation Design

Based on `docs/spec.md` and `docs/interface-stt.md`. Developed on Mac (no CUDA); integration tests with real Qwen3 backend deferred to GPU environment.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UUID version | v7, no hyphens (32 hex chars) | Time-ordered, better for log sorting |
| Mock backend | Fixture-based replay | Realistic progressive partials, test full pipeline on Mac |
| Logging | structlog (JSON) | Battle-tested structured logging with context binding |
| Metrics | prometheus_client | Industry standard, lightweight |
| in_audio_queue bridge | janus queue | Sync get in worker thread, async put in recv_pump — clean dual interface |
| out_event_queue bridge | asyncio.Queue + loop.call_soon_threadsafe | Worker thread puts via threadsafe call, send_pump gets async |

---

## Project Structure

```
realtime-stt-ws/
├── pyproject.toml
├── app.py                          # FastAPI app factory + lifespan
├── config.py                       # Settings via pydantic-settings
├── backends/
│   ├── __init__.py
│   ├── base.py                     # ASRBackend protocol + ASRResult dataclass
│   ├── registry.py                 # Backend registry & create_backend()
│   ├── qwen3.py                    # Qwen3Backend (vLLM, skeleton)
│   └── mock.py                     # MockBackend (fixture-based replay)
├── transport/
│   ├── __init__.py
│   └── ws.py                       # WS endpoint, recv/send pumps, lifecycle
├── protocol/
│   ├── __init__.py
│   └── schema.py                   # Query param validation, message builders
├── session/
│   ├── __init__.py
│   └── state.py                    # Per-connection state, segment_seq, timers
├── inference/
│   ├── __init__.py
│   └── worker.py                   # Thread worker, partial_ms throttle, queue bridge
├── observability/
│   ├── __init__.py
│   ├── logging.py                  # structlog JSON setup
│   └── metrics.py                  # Prometheus counters/histograms
├── references/                     # Test fixtures (existing)
│   ├── zero_shot_prompt.wav
│   └── zero_shot_prompt.txt
├── tests/
│   ├── conftest.py                 # Server fixture (uvicorn in background thread)
│   ├── test_integration.py         # 4 integration tests from spec
│   └── test_unit.py                # Unit tests for schema, session, mock backend
└── docs/
    ├── spec.md
    └── interface-stt.md
```

---

## Dependencies

**Runtime:**
- `fastapi[standard]` (includes uvicorn)
- `structlog`
- `prometheus_client`
- `pydantic-settings`
- `janus` (dual sync/async queue)
- `uuid-utils` (UUID v7)

**Dev:**
- `pytest`
- `pytest-asyncio`
- `websockets` (test WS client)

**Backend-specific (not top-level):**
- `vllm` — imported lazily inside `qwen3.py` only

---

## ASRBackend Protocol & MockBackend

### `backends/base.py`

Exact protocol from spec: `ASRResult` dataclass (`text`, `is_partial`, `is_endpoint`), `ASRBackend` Protocol with 7 methods (`configure`, `push_audio`, `get_partial`, `finalize`, `detect_endpoint`, `reset_segment`, `close`).

### `backends/mock.py` — Fixture-Based Replay

- `__init__(fixture_dir="references")`: loads target text from `zero_shot_prompt.txt`, reads wav header to determine total expected PCM bytes
- `push_audio(pcm_data)`: accumulates byte count
- `get_partial()`: maps `accumulated_bytes / total_bytes` ratio to character slice of target text. Sleeps `MOCK_INFER_MS` (env, default 5ms). Returns `ASRResult` or `None` if empty
- `detect_endpoint()`: returns `True` when bytes >= 95% of total expected, fires once
- `finalize()`: returns full target text, resets byte counter
- `reset_segment()`: clears accumulated bytes and endpoint flag
- `close()`: no-op

### `backends/qwen3.py` — Skeleton

All methods raise `NotImplementedError` except `configure()`/`close()`. Imports `vllm` lazily so Mac installs don't break.

### `backends/registry.py`

```python
_BACKENDS = {
    "qwen3": "backends.qwen3.Qwen3Backend",
    "mock":  "backends.mock.MockBackend",
}
```

---

## Transport & Connection Lifecycle

### `transport/ws.py`

Single endpoint: `WebSocket /v1/stt/ws`

**Connection setup:**
1. Validate query params → `ConnParams` or `error(1008)` + close
2. Generate `conn_id` (uuid7 hex)
3. Create `SessionState`, backend instance, both queues
4. `backend.configure(sample_rate, language, hotwords)`
5. Send `ready`
6. Launch 3 concurrent tasks

**`ws_recv_pump` (async):**
- Binary → `in_audio_queue.put_nowait()`. QueueFull → enqueue `error(1008, "audio queue overflow")` → break
- Text → parse JSON. Only `{"type":"commit"}` → put `COMMIT` sentinel. Else → `error(1008)` → break
- WS disconnect → put `STOP` sentinel → break

**`ws_send_pump` (async):**
- Loop `out_event_queue.get()` → `websocket.send_json()`
- On `STOP` sentinel or error type → send then break

**Cleanup:**
- `asyncio.gather()` with `return_exceptions=True`
- Finally: cancel tasks, `backend.close()`, close WS

**Queue sizes (configurable via config.py):**
- `in_audio_queue`: maxsize=200 (default)
- `out_event_queue`: maxsize=50 (default)

---

## Inference Worker

### `inference/worker.py`

Runs via `asyncio.to_thread()`. Uses janus sync side for `in_audio_queue`.

**Worker loop:**
1. `sync_q.get(timeout=0.05)` — drain items
2. Audio bytes → `backend.push_audio()`
3. `COMMIT` → `backend.finalize()` → emit `final(reason=commit)` → `backend.reset_segment()` → `session.increment_segment()`
4. `STOP` → exit
5. After drain: `backend.detect_endpoint()` → if True → `backend.finalize()` → emit `final(reason=endpoint)` → `backend.reset_segment()` → `session.increment_segment()`
6. If no endpoint: check `partial_ms` throttle → `backend.get_partial()` → emit `partial` if non-empty text

**Thread → asyncio bridge:**
- `loop.call_soon_threadsafe(out_event_queue.put_nowait, event_dict)`
- QueueFull on out_queue → emit `error(1011)`, signal shutdown

---

## Protocol Schema & Session State

### `protocol/schema.py`

- `validate_query_params()` → `ConnParams` (frozen dataclass) or error
- Validates: `call_id` required, `sample_rate` must be 16000, `codec` must be `pcm_s16le`, `channels` must be 1
- Defaults: `frame_ms=20`, `partial_ms=500`, `language="auto"`, `hotwords=None`
- Message builders: `build_ready()`, `build_partial()`, `build_final()`, `build_error()` — all include `created_at` as `time.time()`

### `session/state.py`

```python
class SessionState:
    conn_id: str                        # uuid7 hex
    call_id: str
    segment_seq: int = 1
    segment_start_ts: float             # monotonic
    first_audio_ts: float | None
    first_partial_ts: float | None
    last_partial_ts: float = 0.0
    segment_bytes: int = 0
```

- `increment_segment()`: bump seq, reset timestamps and byte count
- `compute_timing()`: calculate `first_token_ms`, `latency_ms`, `segment_duration_ms`

---

## Observability

### `observability/logging.py`

- structlog with JSON renderer, ISO8601 timestamps
- `get_logger(call_id, conn_id)` returns BoundLogger with pre-bound context
- All events use spec's stable names: `ws.accepted`, `ws.ready_sent`, `ws.closed`, `ws.recv.binary`, `ws.recv.commit`, `ws.send.partial`, `ws.send.final`, `ws.send.error`, `queue.in.overflow`, `queue.out.overflow`, `infer.asr`
- Troubleshoot bundle on non-1000 close: ring buffer of last 20 events, peak queue depths, overflow counts, last inference duration, finalize_reason, close_code

### `observability/metrics.py`

| Category | Metric | Type |
|----------|--------|------|
| WS | `stt_active_connections` | Gauge |
| WS | `stt_close_total{code}` | Counter |
| WS | `stt_error_total{code}` | Counter |
| WS | `stt_in_bytes_total` | Counter |
| WS | `stt_out_events_total{type}` | Counter |
| Queue | `stt_q_in_depth` | Gauge |
| Queue | `stt_q_out_depth` | Gauge |
| Queue | `stt_q_in_overflow_total` | Counter |
| Queue | `stt_q_out_overflow_total` | Counter |
| Infer | `stt_asr_infer_ms` | Histogram |
| Infer | `stt_ttfb_ms` | Histogram |
| Infer | `stt_final_latency_ms` | Histogram |
| Segment | `stt_finalize_total{reason}` | Counter |

Exposed at `GET /metrics` via `prometheus_client.make_asgi_app()`.

---

## Testing (Mac-side, STT_BACKEND=mock)

### Infrastructure

- `conftest.py`: starts FastAPI app via `uvicorn.Server` in background thread, random port, yields WS URL, shuts down after test
- WS client: `websockets` library

### Integration Tests

1. **Happy path**: connect → `ready` → stream wav as 100ms chunks → assert `final` → compare text (strip + collapse whitespace)
2. **Invalid params**: `sample_rate=8000` → assert `error(1008)` → WS closed
3. **Back-pressure**: blast frames without sleep → assert `error(1008)` with "overflow" → WS closed
4. **Commit flow**: send ~0.5s audio → `{"type":"commit"}` → assert `final(seq=1)` → send more audio → commit → assert `final(seq=2)`

### Unit Tests

- `protocol/schema.py`: valid/invalid query param combinations
- `session/state.py`: segment_seq increment, timing computation
- `backends/mock.py`: progressive partial text, endpoint detection threshold, finalize returns full text

---

## One-Liner Commands

```bash
# Install dependencies
uv sync

# Start server (mock backend, local dev)
STT_BACKEND=mock uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Start server (qwen3 backend, production)
STT_BACKEND=qwen3 uv run uvicorn app:app --host 0.0.0.0 --port 8000

# Run tests
STT_BACKEND=mock uv run pytest tests/ -v
```
