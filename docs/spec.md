# realtime-stt-ws — Implementation Spec

A model-agnostic Realtime STT WebSocket Server. It strictly follows the lightweight WebSocket protocol defined in `interface-stt.md`, and provides reproducible integration tests.

**Extensibility model:** The server defines a pluggable `ASRBackend` interface. Adding a new model (whisper / funasr / nemo / …) means adding one backend file — no changes to protocol, routing, or observability.

**Default backend:** `qwen3` — Qwen3-ASR-1.7B via vLLM.

**Core objective:** Inference must never block the WebSocket coroutine (thread worker + asyncio queue + back-pressure), with built-in observability for troubleshooting.

---

## 0. Environment & Resources

### 0.1 Fixed Infrastructure (Do Not Modify)

| Item | Value |
|------|-------|
| Python | 3.12 |
| Package manager | uv |
| FastAPI | 0.128.6 |

### 0.2 Backend: `qwen3` (Default, Phase 1)

| Item | Value |
|------|-------|
| Inference runtime | vllm 0.15.1 |
| Model | Qwen3-ASR-1.7B |
| Implementation | `backends/qwen3.py` → `Qwen3Backend(ASRBackend)` |

### 0.3 Test Fixtures

| Item | Value |
|------|-------|
| Integration test wav | `references/zero_shot_prompt.wav` |
| Integration test expected text | `references/zero_shot_prompt.txt` |

---

## 1. Protocol (Single Source of Truth: `interface-stt.md`)

WebSocket only:

- **Endpoint:** `WebSocket /v1/stt/ws`
- **Query parameters:**
  - `call_id` — required, used for tracing/logs
  - `sample_rate` — required, defaults to 16000
  - `codec` — defaults to `pcm_s16le`, the only supported value
  - `channels` — defaults to 1, the only supported value
  - `frame_ms` — defaults to 20 (client frame duration hint)
  - `partial_ms` — defaults to 500 (partial result throttle interval)
  - `language` — optional, BCP-47 language tag (e.g., `en`, `zh`, `ja`); defaults to `auto` (backend auto-detection)
  - `hotwords` — optional, URL-encoded comma-separated word list (for recognition bias)

### Message Format

**Client → Server:**

- **Binary frames:** Raw PCM16LE audio bytes
- **Text frames (JSON):** Only `{"type":"commit"}` is allowed, requesting immediate finalization of the current utterance (any other JSON is treated as an invalid parameter)

**Server → Client (JSON):**

| Type | Description |
|------|-------------|
| `ready` | Sent immediately after the connection is accepted |
| `partial` | Sent throttled by `partial_ms` |
| `final` | Sent immediately on endpoint detection or commit; includes `first_token_ms` / `latency_ms` / `segment_duration_ms` |
| `error` | Connection must be closed after sending |

**Close codes:**

| Code | Semantics |
|------|-----------|
| 1000 | Normal closure |
| 1008 | Policy violation (invalid parameters) |
| 1011 | Internal server error |

---

## 2. ASR Backend Abstraction

### 2.1 `ASRBackend` Protocol

The inference worker operates against an abstract `ASRBackend` interface. All model-specific logic lives inside the backend implementation — the rest of the server never imports a concrete model.

```python
from typing import Protocol

class ASRResult:
    text: str               # Recognized text
    is_partial: bool        # True for partial, False for final
    is_endpoint: bool       # True if backend detected utterance boundary

class ASRBackend(Protocol):
    """Abstract interface that every ASR backend must implement."""

    def configure(self, sample_rate: int, language: str = "auto", hotwords: list[str] | None = None) -> None:
        """Called once per connection with session parameters."""
        ...

    def push_audio(self, pcm_data: bytes) -> None:
        """Feed audio chunk to the backend's internal buffer."""
        ...

    def get_partial(self) -> ASRResult | None:
        """Run partial inference on buffered audio. Returns None if no meaningful result."""
        ...

    def finalize(self) -> ASRResult:
        """Force-finalize current segment, return result, and reset internal buffer."""
        ...

    def detect_endpoint(self) -> bool:
        """Check if the backend detects an utterance boundary (VAD/energy). Called by the worker."""
        ...

    def reset_segment(self) -> None:
        """Reset internal state for the next segment (called after finalize)."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...
```

### 2.2 Backend Responsibilities vs Worker Responsibilities

| Concern | Owner | Rationale |
|---------|-------|-----------|
| Language mapping (BCP-47 → native format) | Backend | Each model has its own language codes; backend translates |
| Audio buffering & segmentation | Backend | Model-specific buffer sizes and windowing |
| VAD / endpoint detection | Backend | Some models have built-in VAD; others need external |
| Inference (partial & final) | Backend | Model-specific API calls |
| Thread scheduling & lifecycle | Worker | Transport concern, backend-agnostic |
| Queue I/O & back-pressure | Worker | Transport concern, backend-agnostic |
| `partial_ms` throttle | Worker | Protocol concern, backend-agnostic |
| `segment_seq` tracking | Session | Session concern, backend-agnostic |

### 2.3 Backend Registry

Backend is selected by the `STT_BACKEND` env var (default: `qwen3`). The worker receives a factory or instance — it never imports a concrete backend directly.

```python
# backends/registry.py
_BACKENDS = {
    "qwen3":   "backends.qwen3.Qwen3Backend",
    # future:
    # "whisper": "backends.whisper.WhisperBackend",
    # "funasr":  "backends.funasr.FunASRBackend",
    # "nemo":    "backends.nemo.NemoBackend",
}

def create_backend(name: str = "qwen3", **kwargs) -> ASRBackend:
    import importlib
    module_path, class_name = _BACKENDS[name].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)
```

### 2.4 Adding a New Backend (Checklist)

1. Create `backends/<name>.py` implementing `ASRBackend`
2. Register it in `_BACKENDS` dict in `backends/registry.py`
3. Set `STT_BACKEND=<name>` in env / config
4. Done — no changes to transport, protocol, session, worker, or observability

---

## 3. Non-Blocking Architecture: Thread Worker + asyncio Queue + Back-Pressure

**Goal:** The WebSocket handler only performs I/O and lightweight validation. All VAD/ASR inference must never block the event loop.

### 3.1 Three Pumps (Must Be Separate)

#### 1) `ws_recv_pump` (async)

- Receives WS messages:
  - binary → enqueue to `in_audio_queue`
  - text → parse JSON: only `{"type":"commit"}` is allowed; otherwise send `error(code=1008)` and close
- Only fast-path validation (schema/type/state checks) is permitted; no inference or heavy computation

#### 2) `ws_send_pump` (async)

- Dequeues event dicts from `out_event_queue` and calls `send_json`
- No complex logic — avoid making the send pump a bottleneck

#### 3) `inference_worker` (thread)

- Exclusively handles inference and segmentation:
  - Consumes audio frames and commit signals from `in_audio_queue`
  - Calls `backend.push_audio()` for audio frames
  - Calls `backend.detect_endpoint()` to check for utterance boundaries
  - Calls `backend.get_partial()` / `backend.finalize()` for results
  - Produces partial / final events into `out_event_queue` (via thread-safe bridge to the asyncio loop)
- **No direct `await` or direct websocket operations inside the thread**

### 3.2 Queues & Back-Pressure (Required and Testable)

- `in_audio_queue = asyncio.Queue(maxsize=N)` (N must be finite)
- `out_event_queue = asyncio.Queue(maxsize=M)` (M must be finite)

**Default back-pressure policy (must implement as specified):**

| Queue | Behavior when full |
|-------|--------------------|
| `in_audio_queue` | Immediately send `error(code=1008)` with a message clearly stating queue overflow, then close the connection (prevent OOM) |
| `out_event_queue` | Treat as "client consuming too slowly" or "system congestion"; send `error(code=1011)` and close the connection (prevent blocking buildup) |

**Critical event priority:**

`ready` / `error` / `final` must be delivered with best effort (reserving `out_queue` slots or using a dedicated channel for critical events is allowed, but cross-thread direct sends are not).

### 3.3 Thread ↔ asyncio Bridge (Must Be Correct)

The worker thread must use a thread-safe bridge to put events into `out_event_queue`:

- **Allowed:** `loop.call_soon_threadsafe(...)` / `asyncio.run_coroutine_threadsafe(...)` / "thread-safe queue + async pump"
- **Forbidden:** Direct `await` or direct websocket send calls from the worker thread

### 3.4 Lifecycle & Cleanup

- On WS disconnect / exception / server-sent error: must send a stop signal to the worker; the worker must exit promptly; background tasks must be cancelled; pytest must not hang
- Must call `backend.close()` on cleanup
- **Commit semantics:**
  - On receiving commit → call `backend.finalize()` → send final
  - After finalization, call `backend.reset_segment()` and allow the next segment (multi-utterance support)

---

## 4. `segment_seq` Rules (Observability & Troubleshooting)

- `segment_seq` starts at **1**
- Increments only when a `final` is sent: `final` marks the utterance boundary
- `partial` must be tagged with the current `segment_seq` (at minimum in logs and metrics labels; adding it to the message payload is acceptable as long as it does not break protocol field constraints — unless `interface-stt.md` explicitly allows extended fields. **Default: add to logs/metrics only, not to the message**)
- Must record `finalize_reason`: `endpoint` | `commit`

---

## 5. Observability (Built-in, for Troubleshooting)

**Goal:** Any issue can be traced end-to-end via `call_id` through the full chain: connection → queue → inference → send → close.

### 5.1 Structured Logging (JSON)

All log entries must include:

- `call_id`, `conn_id` (per-connection UUID), `segment_seq`
- `event` (stable enum)
- `ts_ms` (millisecond timestamp)

**Required log events:**

| Phase | Events |
|-------|--------|
| Connection | `ws.accepted`, `ws.ready_sent`, `ws.closed{code,reason}` |
| Receive | `ws.recv.binary{bytes}`, `ws.recv.commit` |
| Send | `ws.send.partial`, `ws.send.final{first_token_ms,latency_ms,segment_duration_ms}`, `ws.send.error{code}` |
| Queue | `queue.in.size`, `queue.out.size`, `queue.in.overflow`, `queue.out.overflow` |
| Inference | `infer.asr{infer_ms,audio_ms,text_len}`, (optional) `infer.vad{infer_ms,decision}` |

**troubleshoot_bundle (on abnormal close):**

On non-1000 close or error send, emit a troubleshoot_bundle (in a single log entry):

- Last N event types, peak queue depths, overflow counts, last inference duration, `finalize_reason`, `close_code`

### 5.2 Metrics (Prometheus Recommended; Minimum Set Required)

Naming is up to you, but must be clear and stable:

| Category | Metrics |
|----------|---------|
| ws | `active_connections`, `close_total{code}`, `error_total{code}`, `in_bytes_total`, `out_events_total{type}` |
| queue | `q_in_depth`, `q_out_depth`, `q_in_overflow_total`, `q_out_overflow_total` |
| infer | `asr_infer_ms` (histogram), `ttfb_ms` (histogram), `final_latency_ms` (histogram) |
| segment | `finalize_total{reason}` |

### 5.3 Tracing (Optional Bonus, Not Required)

If implementing OTel tracing:

- Spans must cover at minimum: `recv → enqueue → worker dequeue → asr.run → out enqueue → send`

---

## 6. Endpoint Detection Strategy

Keep it simple, configurable, and testable:

- Server-side endpoint detection enabled by default (delegated to `backend.detect_endpoint()`)
- Client commit must trigger final immediately via `backend.finalize()` (do not wait for endpoint)
- Partial output must obey `partial_ms` throttle (default 500ms); avoid "partial storms" that clog `out_queue`

---

## 7. Integration Tests

Use pytest. Start a real server and run full flows with a WS client. Must validate against the specified wav/txt and cover back-pressure, invalid parameters, and commit.

### 1) Happy Path

1. Connect to `ws://localhost:8000/v1/stt/ws?call_id=test-001&sample_rate=16000`
2. Wait for `ready`
3. Send `zero_shot_prompt.wav` decoded as PCM16LE/16kHz/mono binary frames (recommended: 100ms chunks)
4. Must receive a `final`
5. Compare `final.text` against `zero_shot_prompt.txt` (fixed normalization allowed: trim + collapse whitespace; document the rules)

### 2) Invalid Parameters

- Connecting with `sample_rate=8000` must result in `error(code=1008)` followed by connection close

### 3) Back-Pressure Trigger

- Send binary frames at maximum speed to trigger `in_audio_queue` overflow
- Assert: server sends `error(code=1008, message contains "overflow")` and closes; logs/metrics record the overflow

### 4) Commit Behavior

- Send partial audio → send `{"type":"commit"}` → immediately receive `final`
- After finalization, continue sending audio for the next segment; confirm `segment_seq` increments on the second `final` (assert via logs or metrics)

---

## 8. Project Structure

```
backends/
  base.py              — ASRBackend protocol + ASRResult dataclass
  registry.py          — Backend registry & factory (create_backend)
  qwen3.py             — [default] Qwen3Backend — vLLM + Qwen3-ASR-1.7B
  # whisper.py         — (future) WhisperBackend
  # funasr.py          — (future) FunASRBackend
  # nemo.py            — (future) NemoBackend
transport/
  ws.py                — WS endpoint + recv/send pump + lifecycle
protocol/
  schema.py            — Query param validation, JSON message validation
session/
  state.py             — Per-connection session state, segment_seq, timers
inference/
  worker.py            — Thread worker, partial_ms throttle, out_queue bridge (backend-agnostic)
observability/
  logging.py           — Structured logging
  metrics.py           — Metrics
tests/                 — Integration tests
```

**Boundary rule:** Everything outside `backends/` is backend-agnostic. A new model touches only `backends/`.

---

## 9. One-Liner Commands (Must Be Copy-Pasteable)

Provide:

- `uv` sync dependencies / create environment
- Start server (dev mode, `STT_BACKEND=qwen3` by default)
- Run pytest integration tests

---

## 10. Pre-Delivery Checklist

- [ ] Strict alignment with `interface-stt.md`: `/v1/stt/ws`, query params, ready/partial/final/error, binary audio, commit, 1000/1008/1011
- [ ] `ASRBackend` protocol defined; inference worker is backend-agnostic; `Qwen3Backend` is the only model-specific code
- [ ] WS handler runs no inference; inference runs in worker thread; bridge is correct (no cross-thread await / no event loop blocking)
- [ ] in/out queues have finite maxsize; back-pressure policy implemented per spec; overflow covered by tests
- [ ] Partial obeys `partial_ms`; final includes `first_token_ms` / `latency_ms` / `segment_duration_ms`
- [ ] `segment_seq` increments only with final; `finalize_reason` is recorded
- [ ] Structured logging + minimum metrics complete; end-to-end troubleshooting possible via `call_id`
- [ ] Integration tests pass with the specified wav/txt; invalid parameters, back-pressure, and commit are covered
- [ ] One-liner commands for uv + server start + test run, plus a troubleshooting guide
- [ ] Adding a new backend = one file in `backends/` + registry entry + env var; zero changes elsewhere
