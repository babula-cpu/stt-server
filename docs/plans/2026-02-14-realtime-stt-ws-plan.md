# realtime-stt-ws Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a model-agnostic Realtime STT WebSocket server with a pluggable backend interface, mock backend for Mac-side testing, and full observability.

**Architecture:** FastAPI WebSocket endpoint with 3-pump design (recv_pump, send_pump, inference_worker thread). janus queue bridges async recv to sync worker thread. Backend abstraction isolates all model-specific code.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, structlog, prometheus_client, janus, uuid-utils, pydantic-settings

**Important context:**
- Reference wav is float32/24kHz (NOT PCM16LE/16kHz) — tests must convert
- Wav duration: 3.48s (83520 samples at 24kHz, 334080 data bytes)
- Expected PCM16LE 16kHz: 55680 samples, 111360 bytes
- Reference text: `希望你以后能够做的比我还好哟`
- All UUIDs: v7, no hyphens (32 hex chars)

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: all `__init__.py` files
- Run: `git init`, `uv sync`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "realtime-stt-ws"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi[standard]>=0.128.6,<1",
    "structlog>=24.0",
    "prometheus-client>=0.21",
    "pydantic-settings>=2.0",
    "janus>=1.0",
    "uuid-utils>=0.9",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "websockets>=13.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**Step 2: Create directory structure and __init__.py files**

Create empty `__init__.py` in: `backends/`, `transport/`, `protocol/`, `session/`, `inference/`, `observability/`, `tests/`

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
.DS_Store
```

**Step 4: git init and install deps**

```bash
cd /Users/rasonyang/workspaces/vllm/realtime-stt-ws
git init
uv sync
```

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore backends/__init__.py transport/__init__.py protocol/__init__.py session/__init__.py inference/__init__.py observability/__init__.py tests/__init__.py docs/ references/
git commit -m "chore: project scaffold with dependencies and directory structure"
```

---

## Task 2: config.py

**Files:**
- Create: `config.py`

**Step 1: Write config**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "STT_"}

    backend: str = "qwen3"
    host: str = "0.0.0.0"
    port: int = 8000

    # Queue sizes
    in_queue_size: int = 200
    out_queue_size: int = 50

    # Supported audio params (for validation)
    supported_sample_rates: list[int] = [16000]
    supported_codecs: list[str] = ["pcm_s16le"]
    supported_channels: list[int] = [1]


settings = Settings()
```

**Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add config module with pydantic-settings"
```

---

## Task 3: backends/base.py

**Files:**
- Create: `backends/base.py`

**Step 1: Write ASRBackend protocol and ASRResult**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ASRResult:
    text: str
    is_partial: bool
    is_endpoint: bool


class ASRBackend(Protocol):
    """Abstract interface that every ASR backend must implement."""

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
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
        """Check if the backend detects an utterance boundary."""
        ...

    def reset_segment(self) -> None:
        """Reset internal state for the next segment."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...
```

**Step 2: Commit**

```bash
git add backends/base.py
git commit -m "feat: define ASRBackend protocol and ASRResult dataclass"
```

---

## Task 4: protocol/schema.py + unit tests

**Files:**
- Create: `protocol/schema.py`
- Create: `tests/test_schema.py`

**Step 1: Write tests**

```python
import pytest
import time
from protocol.schema import validate_query_params, build_ready, build_partial, build_final, build_error, ConnParams, ValidationError


class TestValidateQueryParams:
    def test_valid_minimal(self):
        params = validate_query_params({"call_id": "test-001", "sample_rate": "16000"})
        assert params.call_id == "test-001"
        assert params.sample_rate == 16000
        assert params.codec == "pcm_s16le"
        assert params.channels == 1
        assert params.frame_ms == 20
        assert params.partial_ms == 500
        assert params.language == "auto"
        assert params.hotwords is None

    def test_valid_all_params(self):
        params = validate_query_params({
            "call_id": "test-002",
            "sample_rate": "16000",
            "codec": "pcm_s16le",
            "channels": "1",
            "frame_ms": "30",
            "partial_ms": "200",
            "language": "zh",
            "hotwords": "hello%20world%2Cfoo",
        })
        assert params.partial_ms == 200
        assert params.language == "zh"
        assert params.hotwords == ["hello world", "foo"]

    def test_missing_call_id(self):
        with pytest.raises(ValidationError, match="call_id"):
            validate_query_params({"sample_rate": "16000"})

    def test_missing_sample_rate(self):
        with pytest.raises(ValidationError, match="sample_rate"):
            validate_query_params({"call_id": "test"})

    def test_invalid_sample_rate(self):
        with pytest.raises(ValidationError, match="16000"):
            validate_query_params({"call_id": "test", "sample_rate": "8000"})

    def test_invalid_codec(self):
        with pytest.raises(ValidationError, match="pcm_s16le"):
            validate_query_params({"call_id": "test", "sample_rate": "16000", "codec": "opus"})

    def test_invalid_channels(self):
        with pytest.raises(ValidationError, match="1"):
            validate_query_params({"call_id": "test", "sample_rate": "16000", "channels": "2"})


class TestMessageBuilders:
    def test_build_ready(self):
        params = ConnParams(
            call_id="test-001", sample_rate=16000, codec="pcm_s16le",
            channels=1, frame_ms=20, partial_ms=500, language="en", hotwords=None,
        )
        msg = build_ready(params)
        assert msg["type"] == "ready"
        assert msg["call_id"] == "test-001"
        assert msg["sample_rate"] == 16000
        assert msg["language"] == "en"
        assert "created_at" in msg

    def test_build_partial(self):
        msg = build_partial("test-001", 1, "hello")
        assert msg["type"] == "partial"
        assert msg["segment_seq"] == 1
        assert msg["text"] == "hello"
        assert msg["final"] is False

    def test_build_final(self):
        msg = build_final("test-001", 1, "hello world", 85, 120, 2000)
        assert msg["type"] == "final"
        assert msg["final"] is True
        assert msg["first_token_ms"] == 85
        assert msg["latency_ms"] == 120
        assert msg["segment_duration_ms"] == 2000

    def test_build_error(self):
        msg = build_error("test-001", 1008, "invalid param")
        assert msg["type"] == "error"
        assert msg["code"] == 1008
        assert msg["message"] == "invalid param"
```

**Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock uv run pytest tests/test_schema.py -v
```
Expected: FAIL (import errors)

**Step 3: Write protocol/schema.py**

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import unquote


class ValidationError(Exception):
    def __init__(self, message: str, code: int = 1008):
        self.message = message
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ConnParams:
    call_id: str
    sample_rate: int
    codec: str
    channels: int
    frame_ms: int
    partial_ms: int
    language: str
    hotwords: list[str] | None


def validate_query_params(params: dict[str, str]) -> ConnParams:
    call_id = params.get("call_id")
    if not call_id:
        raise ValidationError("call_id is required")

    raw_sr = params.get("sample_rate")
    if not raw_sr:
        raise ValidationError("sample_rate is required")
    sample_rate = int(raw_sr)
    if sample_rate != 16000:
        raise ValidationError(f"Invalid sample_rate: {sample_rate}. Must be 16000.")

    codec = params.get("codec", "pcm_s16le")
    if codec != "pcm_s16le":
        raise ValidationError(f"Invalid codec: {codec}. Must be pcm_s16le.")

    channels = int(params.get("channels", "1"))
    if channels != 1:
        raise ValidationError(f"Invalid channels: {channels}. Must be 1.")

    frame_ms = int(params.get("frame_ms", "20"))
    partial_ms = int(params.get("partial_ms", "500"))
    language = params.get("language", "auto")

    hotwords = None
    raw_hotwords = params.get("hotwords")
    if raw_hotwords:
        decoded = unquote(raw_hotwords)
        hotwords = [w.strip() for w in decoded.split(",") if w.strip()]

    return ConnParams(
        call_id=call_id,
        sample_rate=sample_rate,
        codec=codec,
        channels=channels,
        frame_ms=frame_ms,
        partial_ms=partial_ms,
        language=language,
        hotwords=hotwords,
    )


def _now() -> float:
    return time.time()


def build_ready(params: ConnParams) -> dict:
    return {
        "type": "ready",
        "call_id": params.call_id,
        "created_at": _now(),
        "sample_rate": params.sample_rate,
        "codec": params.codec,
        "channels": params.channels,
        "frame_ms": params.frame_ms,
        "partial_ms": params.partial_ms,
        "language": params.language,
    }


def build_partial(call_id: str, segment_seq: int, text: str) -> dict:
    return {
        "type": "partial",
        "call_id": call_id,
        "created_at": _now(),
        "segment_seq": segment_seq,
        "text": text,
        "final": False,
    }


def build_final(
    call_id: str,
    segment_seq: int,
    text: str,
    first_token_ms: int,
    latency_ms: int,
    segment_duration_ms: int,
) -> dict:
    return {
        "type": "final",
        "call_id": call_id,
        "created_at": _now(),
        "segment_seq": segment_seq,
        "text": text,
        "final": True,
        "first_token_ms": first_token_ms,
        "latency_ms": latency_ms,
        "segment_duration_ms": segment_duration_ms,
    }


def build_error(call_id: str, code: int, message: str) -> dict:
    return {
        "type": "error",
        "call_id": call_id,
        "created_at": _now(),
        "code": code,
        "message": message,
    }
```

**Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock uv run pytest tests/test_schema.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add protocol/schema.py tests/test_schema.py
git commit -m "feat: protocol schema validation and message builders with tests"
```

---

## Task 5: session/state.py + unit tests

**Files:**
- Create: `session/state.py`
- Create: `tests/test_session.py`

**Step 1: Write tests**

```python
import time
from session.state import SessionState


class TestSessionState:
    def test_initial_state(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        assert s.segment_seq == 1
        assert s.segment_bytes == 0
        assert s.first_audio_ts is None
        assert s.first_partial_ts is None

    def test_record_audio(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        s.record_audio(3200)
        assert s.segment_bytes == 3200
        assert s.first_audio_ts is not None
        first = s.first_audio_ts
        s.record_audio(3200)
        assert s.segment_bytes == 6400
        assert s.first_audio_ts == first  # unchanged

    def test_record_partial(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        s.record_partial()
        assert s.first_partial_ts is not None
        assert s.last_partial_ts > 0

    def test_should_send_partial(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        s.partial_ms = 500
        # First partial always allowed
        assert s.should_send_partial() is True
        s.record_partial()
        # Immediately after, throttled
        assert s.should_send_partial() is False

    def test_increment_segment(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        s.record_audio(3200)
        s.record_partial()
        s.increment_segment()
        assert s.segment_seq == 2
        assert s.segment_bytes == 0
        assert s.first_audio_ts is None
        assert s.first_partial_ts is None

    def test_compute_timing(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        s.sample_rate = 16000
        s.record_audio(32000)  # 1 second of audio
        # Simulate some passage of time by setting timestamps directly
        s.first_audio_ts = time.monotonic() - 0.5
        s.first_partial_ts = s.first_audio_ts + 0.085
        timing = s.compute_timing()
        assert timing["segment_duration_ms"] == 1000  # 32000 bytes / 2 / 16000 * 1000
        assert timing["first_token_ms"] == 85
        assert "latency_ms" in timing

    def test_segment_duration_from_bytes(self):
        s = SessionState(conn_id="abc123", call_id="test-001")
        s.sample_rate = 16000
        s.record_audio(64000)  # 2 seconds
        timing = s.compute_timing()
        assert timing["segment_duration_ms"] == 2000
```

**Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock uv run pytest tests/test_session.py -v
```

**Step 3: Write session/state.py**

```python
from __future__ import annotations

import time


class SessionState:
    __slots__ = (
        "conn_id",
        "call_id",
        "sample_rate",
        "partial_ms",
        "segment_seq",
        "segment_start_ts",
        "first_audio_ts",
        "first_partial_ts",
        "last_partial_ts",
        "segment_bytes",
    )

    def __init__(self, conn_id: str, call_id: str) -> None:
        self.conn_id = conn_id
        self.call_id = call_id
        self.sample_rate: int = 16000
        self.partial_ms: int = 500
        self.segment_seq: int = 1
        self.segment_start_ts: float = time.monotonic()
        self.first_audio_ts: float | None = None
        self.first_partial_ts: float | None = None
        self.last_partial_ts: float = 0.0
        self.segment_bytes: int = 0

    def record_audio(self, num_bytes: int) -> None:
        if self.first_audio_ts is None:
            self.first_audio_ts = time.monotonic()
        self.segment_bytes += num_bytes

    def record_partial(self) -> None:
        now = time.monotonic()
        if self.first_partial_ts is None:
            self.first_partial_ts = now
        self.last_partial_ts = now

    def should_send_partial(self) -> bool:
        if self.last_partial_ts == 0.0:
            return True
        elapsed_ms = (time.monotonic() - self.last_partial_ts) * 1000
        return elapsed_ms >= self.partial_ms

    def increment_segment(self) -> None:
        self.segment_seq += 1
        self.segment_start_ts = time.monotonic()
        self.first_audio_ts = None
        self.first_partial_ts = None
        self.last_partial_ts = 0.0
        self.segment_bytes = 0

    def compute_timing(self) -> dict:
        now = time.monotonic()
        segment_duration_ms = int(self.segment_bytes / 2 / self.sample_rate * 1000)

        first_token_ms = 0
        if self.first_audio_ts is not None and self.first_partial_ts is not None:
            first_token_ms = int((self.first_partial_ts - self.first_audio_ts) * 1000)

        latency_ms = 0
        if self.first_audio_ts is not None:
            latency_ms = int((now - self.first_audio_ts) * 1000)

        return {
            "first_token_ms": first_token_ms,
            "latency_ms": latency_ms,
            "segment_duration_ms": segment_duration_ms,
        }
```

**Step 4: Run tests**

```bash
STT_BACKEND=mock uv run pytest tests/test_session.py -v
```

**Step 5: Commit**

```bash
git add session/state.py tests/test_session.py
git commit -m "feat: session state with segment tracking and timing computation"
```

---

## Task 6: observability/logging.py

**Files:**
- Create: `observability/logging.py`

**Step 1: Write structured logging setup**

```python
from __future__ import annotations

import structlog


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(call_id: str, conn_id: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger().bind(call_id=call_id, conn_id=conn_id)


class TroubleshootCollector:
    """Ring buffer collecting recent events for troubleshoot bundle on abnormal close."""

    __slots__ = ("_events", "_max", "_peak_in", "_peak_out", "_overflow_in", "_overflow_out", "_last_infer_ms", "_finalize_reason")

    def __init__(self, max_events: int = 20) -> None:
        self._events: list[str] = []
        self._max = max_events
        self._peak_in: int = 0
        self._peak_out: int = 0
        self._overflow_in: int = 0
        self._overflow_out: int = 0
        self._last_infer_ms: float = 0.0
        self._finalize_reason: str | None = None

    def record_event(self, event: str) -> None:
        self._events.append(event)
        if len(self._events) > self._max:
            self._events.pop(0)

    def record_queue_depth(self, in_depth: int, out_depth: int) -> None:
        self._peak_in = max(self._peak_in, in_depth)
        self._peak_out = max(self._peak_out, out_depth)

    def record_overflow(self, queue: str) -> None:
        if queue == "in":
            self._overflow_in += 1
        else:
            self._overflow_out += 1

    def record_infer(self, infer_ms: float) -> None:
        self._last_infer_ms = infer_ms

    def record_finalize(self, reason: str) -> None:
        self._finalize_reason = reason

    def build_bundle(self, close_code: int) -> dict:
        return {
            "last_events": list(self._events),
            "peak_in_depth": self._peak_in,
            "peak_out_depth": self._peak_out,
            "overflow_in_count": self._overflow_in,
            "overflow_out_count": self._overflow_out,
            "last_infer_ms": self._last_infer_ms,
            "finalize_reason": self._finalize_reason,
            "close_code": close_code,
        }
```

**Step 2: Commit**

```bash
git add observability/logging.py
git commit -m "feat: structured logging with structlog and troubleshoot collector"
```

---

## Task 7: observability/metrics.py

**Files:**
- Create: `observability/metrics.py`

**Step 1: Write Prometheus metrics**

```python
from prometheus_client import Counter, Gauge, Histogram

# WS
active_connections = Gauge("stt_active_connections", "Number of active WS connections")
close_total = Counter("stt_close_total", "Total WS closes", ["code"])
error_total = Counter("stt_error_total", "Total errors sent", ["code"])
in_bytes_total = Counter("stt_in_bytes_total", "Total audio bytes received")
out_events_total = Counter("stt_out_events_total", "Total events sent", ["type"])

# Queue
q_in_depth = Gauge("stt_q_in_depth", "Current input queue depth")
q_out_depth = Gauge("stt_q_out_depth", "Current output queue depth")
q_in_overflow_total = Counter("stt_q_in_overflow_total", "Input queue overflow count")
q_out_overflow_total = Counter("stt_q_out_overflow_total", "Output queue overflow count")

# Inference
asr_infer_ms = Histogram("stt_asr_infer_ms", "ASR inference duration in ms", buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000])
ttfb_ms = Histogram("stt_ttfb_ms", "Time to first token in ms", buckets=[25, 50, 100, 250, 500, 1000, 2500])
final_latency_ms = Histogram("stt_final_latency_ms", "Final result latency in ms", buckets=[50, 100, 250, 500, 1000, 2500, 5000])

# Segment
finalize_total = Counter("stt_finalize_total", "Total finalizations", ["reason"])
```

**Step 2: Commit**

```bash
git add observability/metrics.py
git commit -m "feat: Prometheus metrics definitions"
```

---

## Task 8: backends/mock.py + unit tests

**Files:**
- Create: `backends/mock.py`
- Create: `tests/test_mock_backend.py`

**Step 1: Write tests**

```python
import pytest
from backends.mock import MockBackend


class TestMockBackend:
    def setup_method(self):
        self.backend = MockBackend()
        self.backend.configure(sample_rate=16000, language="auto")

    def test_configure(self):
        assert self.backend._sample_rate == 16000
        assert self.backend._target_text is not None
        assert len(self.backend._target_text) > 0

    def test_push_audio_accumulates(self):
        self.backend.push_audio(b"\x00" * 3200)
        assert self.backend._buf_bytes == 3200
        self.backend.push_audio(b"\x00" * 3200)
        assert self.backend._buf_bytes == 6400

    def test_get_partial_returns_progressive_text(self):
        target = self.backend._target_text
        total = self.backend._total_expected_bytes
        # Push 50% of audio
        self.backend.push_audio(b"\x00" * (total // 2))
        result = self.backend.get_partial()
        assert result is not None
        assert result.is_partial is True
        assert result.is_endpoint is False
        assert len(result.text) > 0
        assert len(result.text) < len(target)

    def test_get_partial_returns_none_with_no_audio(self):
        result = self.backend.get_partial()
        assert result is None

    def test_detect_endpoint_at_threshold(self):
        total = self.backend._total_expected_bytes
        # Below threshold
        self.backend.push_audio(b"\x00" * int(total * 0.9))
        assert self.backend.detect_endpoint() is False
        # At threshold
        self.backend.push_audio(b"\x00" * int(total * 0.1))
        assert self.backend.detect_endpoint() is True
        # Fires only once
        assert self.backend.detect_endpoint() is False

    def test_finalize_returns_full_text(self):
        target = self.backend._target_text
        self.backend.push_audio(b"\x00" * 1000)
        result = self.backend.finalize()
        assert result.text == target
        assert result.is_partial is False
        assert result.is_endpoint is True

    def test_reset_segment_clears_state(self):
        self.backend.push_audio(b"\x00" * 5000)
        self.backend.detect_endpoint()  # may or may not fire
        self.backend.reset_segment()
        assert self.backend._buf_bytes == 0
        assert self.backend._endpoint_fired is False

    def test_close_is_noop(self):
        self.backend.close()  # should not raise
```

**Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock uv run pytest tests/test_mock_backend.py -v
```

**Step 3: Write backends/mock.py**

```python
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
```

**Step 4: Run tests**

```bash
STT_BACKEND=mock uv run pytest tests/test_mock_backend.py -v
```

**Step 5: Commit**

```bash
git add backends/mock.py tests/test_mock_backend.py
git commit -m "feat: fixture-based mock backend with unit tests"
```

---

## Task 9: backends/registry.py + backends/qwen3.py

**Files:**
- Create: `backends/registry.py`
- Create: `backends/qwen3.py`

**Step 1: Write registry**

```python
from __future__ import annotations

import importlib

from backends.base import ASRBackend

_BACKENDS: dict[str, str] = {
    "qwen3": "backends.qwen3.Qwen3Backend",
    "mock": "backends.mock.MockBackend",
}


def create_backend(name: str = "qwen3", **kwargs: object) -> ASRBackend:
    dotted = _BACKENDS.get(name)
    if dotted is None:
        raise ValueError(f"Unknown backend: {name!r}. Available: {sorted(_BACKENDS)}")
    module_path, class_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)
```

**Step 2: Write qwen3 skeleton**

```python
from __future__ import annotations

from backends.base import ASRResult


class Qwen3Backend:
    """Qwen3-ASR-1.7B via vLLM. Requires CUDA — skeleton only on Mac."""

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("Qwen3Backend requires vLLM + CUDA")

    def push_audio(self, pcm_data: bytes) -> None:
        raise NotImplementedError

    def get_partial(self) -> ASRResult | None:
        raise NotImplementedError

    def finalize(self) -> ASRResult:
        raise NotImplementedError

    def detect_endpoint(self) -> bool:
        raise NotImplementedError

    def reset_segment(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass
```

**Step 3: Commit**

```bash
git add backends/registry.py backends/qwen3.py
git commit -m "feat: backend registry and qwen3 skeleton"
```

---

## Task 10: inference/worker.py

**Files:**
- Create: `inference/worker.py`

**Step 1: Write inference worker**

```python
from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any

import janus

from backends.base import ASRBackend, ASRResult
from observability import metrics
from observability.logging import TroubleshootCollector
from protocol.schema import build_error, build_final, build_partial
from session.state import SessionState

# Sentinel objects
COMMIT = object()
STOP = object()

logger = logging.getLogger(__name__)


def run_worker(
    *,
    backend: ASRBackend,
    session: SessionState,
    in_q_sync: janus.SyncQueue[Any],
    out_q: asyncio.Queue[Any],
    loop: asyncio.AbstractEventLoop,
    troubleshoot: TroubleshootCollector,
    log: Any,
) -> None:
    """Inference worker — runs in a dedicated thread. Never touches the event loop directly."""

    def _emit(event: dict) -> None:
        """Thread-safe put into the async out_event_queue."""
        try:
            loop.call_soon_threadsafe(out_q.put_nowait, event)
        except asyncio.QueueFull:
            metrics.q_out_overflow_total.inc()
            troubleshoot.record_overflow("out")
            err = build_error(session.call_id, 1011, "output queue overflow")
            try:
                loop.call_soon_threadsafe(out_q.put_nowait, err)
            except Exception:
                pass
            return
        except RuntimeError:
            # Loop closed
            return

    def _emit_final(result: ASRResult, reason: str) -> None:
        troubleshoot.record_finalize(reason)
        metrics.finalize_total.labels(reason=reason).inc()
        timing = session.compute_timing()
        metrics.final_latency_ms.observe(timing["latency_ms"])
        if timing["first_token_ms"] > 0:
            metrics.ttfb_ms.observe(timing["first_token_ms"])
        event = build_final(
            session.call_id,
            session.segment_seq,
            result.text,
            timing["first_token_ms"],
            timing["latency_ms"],
            timing["segment_duration_ms"],
        )
        _emit(event)
        troubleshoot.record_event("ws.send.final")
        metrics.out_events_total.labels(type="final").inc()
        log.info("ws.send.final", segment_seq=session.segment_seq, reason=reason, **timing)
        session.increment_segment()

    try:
        while True:
            # Drain from input queue
            try:
                item = in_q_sync.get(timeout=0.05)
            except queue.Empty:
                # No items — check endpoint / partial
                if backend.detect_endpoint():
                    t0 = time.monotonic()
                    result = backend.finalize()
                    infer_ms = (time.monotonic() - t0) * 1000
                    metrics.asr_infer_ms.observe(infer_ms)
                    troubleshoot.record_infer(infer_ms)
                    if result.text.strip():
                        _emit_final(result, "endpoint")
                    backend.reset_segment()
                elif session.should_send_partial():
                    t0 = time.monotonic()
                    result = backend.get_partial()
                    infer_ms = (time.monotonic() - t0) * 1000
                    if result is not None and result.text.strip():
                        metrics.asr_infer_ms.observe(infer_ms)
                        troubleshoot.record_infer(infer_ms)
                        troubleshoot.record_event("ws.send.partial")
                        event = build_partial(session.call_id, session.segment_seq, result.text)
                        _emit(event)
                        metrics.out_events_total.labels(type="partial").inc()
                        session.record_partial()
                        log.info("ws.send.partial", segment_seq=session.segment_seq, text_len=len(result.text))
                continue

            if item is STOP:
                break

            if item is COMMIT:
                troubleshoot.record_event("ws.recv.commit")
                log.info("ws.recv.commit", segment_seq=session.segment_seq)
                t0 = time.monotonic()
                result = backend.finalize()
                infer_ms = (time.monotonic() - t0) * 1000
                metrics.asr_infer_ms.observe(infer_ms)
                troubleshoot.record_infer(infer_ms)
                if result.text.strip():
                    _emit_final(result, "commit")
                backend.reset_segment()
                continue

            # Binary audio data
            pcm_data: bytes = item
            backend.push_audio(pcm_data)
            session.record_audio(len(pcm_data))
            metrics.in_bytes_total.inc(len(pcm_data))

            # Update queue depth metrics
            troubleshoot.record_queue_depth(in_q_sync.qsize(), out_q.qsize())
            metrics.q_in_depth.set(in_q_sync.qsize())
            metrics.q_out_depth.set(out_q.qsize())

            # Check endpoint after receiving audio
            if backend.detect_endpoint():
                t0 = time.monotonic()
                result = backend.finalize()
                infer_ms = (time.monotonic() - t0) * 1000
                metrics.asr_infer_ms.observe(infer_ms)
                troubleshoot.record_infer(infer_ms)
                if result.text.strip():
                    _emit_final(result, "endpoint")
                backend.reset_segment()
            elif session.should_send_partial():
                t0 = time.monotonic()
                result = backend.get_partial()
                infer_ms = (time.monotonic() - t0) * 1000
                if result is not None and result.text.strip():
                    metrics.asr_infer_ms.observe(infer_ms)
                    troubleshoot.record_infer(infer_ms)
                    troubleshoot.record_event("ws.send.partial")
                    event = build_partial(session.call_id, session.segment_seq, result.text)
                    _emit(event)
                    metrics.out_events_total.labels(type="partial").inc()
                    session.record_partial()

    except Exception:
        log.exception("infer.worker.error")
    finally:
        # Signal send pump to stop
        try:
            loop.call_soon_threadsafe(out_q.put_nowait, STOP)
        except Exception:
            pass
```

**Step 2: Commit**

```bash
git add inference/worker.py
git commit -m "feat: inference worker with thread-safe queue bridge and back-pressure"
```

---

## Task 11: transport/ws.py

**Files:**
- Create: `transport/ws.py`

**Step 1: Write WebSocket endpoint**

```python
from __future__ import annotations

import asyncio
import json
from typing import Any

import janus
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid_utils import uuid7

from backends.registry import create_backend
from config import settings
from inference.worker import COMMIT, STOP, run_worker
from observability import metrics
from observability.logging import TroubleshootCollector, get_logger
from protocol.schema import (
    ConnParams,
    ValidationError,
    build_error,
    build_ready,
    validate_query_params,
)
from session.state import SessionState

router = APIRouter()


@router.websocket("/v1/stt/ws")
async def stt_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    metrics.active_connections.inc()

    # Parse query params
    params_dict = dict(websocket.query_params)
    call_id = params_dict.get("call_id", "unknown")
    conn_id = uuid7().hex

    log = get_logger(call_id, conn_id)
    log.info("ws.accepted")

    troubleshoot = TroubleshootCollector()
    close_code = 1000

    try:
        try:
            conn_params = validate_query_params(params_dict)
        except ValidationError as e:
            troubleshoot.record_event("ws.send.error")
            err = build_error(call_id, e.code, e.message)
            await websocket.send_json(err)
            metrics.error_total.labels(code=str(e.code)).inc()
            metrics.out_events_total.labels(type="error").inc()
            close_code = e.code
            await websocket.close(code=e.code, reason=e.message)
            return

        # Create backend
        backend = create_backend(settings.backend)
        backend.configure(
            sample_rate=conn_params.sample_rate,
            language=conn_params.language,
            hotwords=conn_params.hotwords,
        )

        # Create session
        session = SessionState(conn_id=conn_id, call_id=conn_params.call_id)
        session.sample_rate = conn_params.sample_rate
        session.partial_ms = conn_params.partial_ms

        # Create queues
        in_janus: janus.Queue[Any] = janus.Queue(maxsize=settings.in_queue_size)
        out_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=settings.out_queue_size)

        # Send ready
        ready_msg = build_ready(conn_params)
        await websocket.send_json(ready_msg)
        troubleshoot.record_event("ws.ready_sent")
        metrics.out_events_total.labels(type="ready").inc()
        log.info("ws.ready_sent")

        loop = asyncio.get_running_loop()

        # Launch pumps + worker
        recv_task = asyncio.create_task(_recv_pump(websocket, in_janus.async_q, out_q, session, log, troubleshoot))
        send_task = asyncio.create_task(_send_pump(websocket, out_q, log, troubleshoot))
        worker_task = asyncio.ensure_future(
            asyncio.to_thread(
                run_worker,
                backend=backend,
                session=session,
                in_q_sync=in_janus.sync_q,
                out_q=out_q,
                loop=loop,
                troubleshoot=troubleshoot,
                log=log,
            )
        )

        try:
            await asyncio.gather(recv_task, send_task, worker_task, return_exceptions=True)
        finally:
            for task in (recv_task, send_task, worker_task):
                if not task.done():
                    task.cancel()
            backend.close()
            in_janus.close()
            await in_janus.wait_closed()

    except WebSocketDisconnect:
        log.info("ws.closed", code=1000, reason="client disconnect")
    except Exception:
        close_code = 1011
        log.exception("ws.error")
    finally:
        metrics.active_connections.dec()
        metrics.close_total.labels(code=str(close_code)).inc()
        if close_code != 1000:
            bundle = troubleshoot.build_bundle(close_code)
            log.warning("ws.troubleshoot_bundle", **bundle)
        log.info("ws.closed", code=close_code)


async def _recv_pump(
    ws: WebSocket,
    in_q: janus.AsyncQueue[Any],
    out_q: asyncio.Queue[Any],
    session: SessionState,
    log: Any,
    troubleshoot: TroubleshootCollector,
) -> None:
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                pcm_data = message["bytes"]
                troubleshoot.record_event("ws.recv.binary")
                try:
                    in_q.put_nowait(pcm_data)
                except asyncio.QueueFull:
                    metrics.q_in_overflow_total.inc()
                    troubleshoot.record_overflow("in")
                    troubleshoot.record_event("queue.in.overflow")
                    log.warning("queue.in.overflow", q_size=in_q.qsize())
                    err = build_error(session.call_id, 1008, "audio queue overflow")
                    try:
                        out_q.put_nowait(err)
                    except asyncio.QueueFull:
                        pass
                    break

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    err = build_error(session.call_id, 1008, "Invalid JSON")
                    try:
                        out_q.put_nowait(err)
                    except asyncio.QueueFull:
                        pass
                    break

                if data.get("type") == "commit":
                    try:
                        in_q.put_nowait(COMMIT)
                    except asyncio.QueueFull:
                        pass
                else:
                    err = build_error(session.call_id, 1008, f"Invalid message type: {data.get('type')}")
                    try:
                        out_q.put_nowait(err)
                    except asyncio.QueueFull:
                        pass
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws.recv_pump.error")
    finally:
        # Signal worker to stop
        try:
            in_q.put_nowait(STOP)
        except (asyncio.QueueFull, janus.AsyncQueueShutDown):
            pass


async def _send_pump(
    ws: WebSocket,
    out_q: asyncio.Queue[Any],
    log: Any,
    troubleshoot: TroubleshootCollector,
) -> None:
    try:
        while True:
            event = await out_q.get()
            if event is STOP:
                break

            if not isinstance(event, dict):
                continue

            msg_type = event.get("type")
            try:
                await ws.send_json(event)
            except Exception:
                break

            if msg_type == "error":
                code = event.get("code", 1011)
                metrics.error_total.labels(code=str(code)).inc()
                metrics.out_events_total.labels(type="error").inc()
                troubleshoot.record_event("ws.send.error")
                try:
                    await ws.close(code=code, reason=event.get("message", ""))
                except Exception:
                    pass
                break
    except Exception:
        log.exception("ws.send_pump.error")
```

**Step 2: Commit**

```bash
git add transport/ws.py
git commit -m "feat: WebSocket transport with recv/send pumps and lifecycle management"
```

---

## Task 12: app.py

**Files:**
- Create: `app.py`

**Step 1: Write app entry point**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from config import settings
from observability.logging import setup_logging
from transport.ws import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(title="realtime-stt-ws", lifespan=lifespan)
app.include_router(router)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.backend}
```

**Step 2: Smoke test — start server**

```bash
STT_BACKEND=mock uv run uvicorn app:app --host 127.0.0.1 --port 8000 &
sleep 2
curl -s http://127.0.0.1:8000/health
# Expected: {"status":"ok","backend":"mock"}
kill %1
```

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: FastAPI app entry point with health check and metrics"
```

---

## Task 13: Test infrastructure + integration tests

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_integration.py`

**Step 1: Write conftest.py with server fixture and wav loader**

```python
from __future__ import annotations

import asyncio
import struct
import threading
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
    import socket

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
    import time

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
```

**Step 2: Write integration tests**

```python
import asyncio
import json
import re

import pytest
import websockets


@pytest.mark.asyncio
async def test_happy_path(server_url: str, pcm_audio: bytes, expected_text: str):
    """Connect, stream wav, receive final, compare text."""
    url = f"{server_url}/v1/stt/ws?call_id=test-happy&sample_rate=16000"
    async with websockets.connect(url) as ws:
        # Wait for ready
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready"
        assert msg["call_id"] == "test-happy"

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
    url = f"{server_url}/v1/stt/ws?call_id=test-invalid&sample_rate=8000"
    async with websockets.connect(url) as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        msg = json.loads(raw)
        assert msg["type"] == "error"
        assert msg["code"] == 1008
        assert "16000" in msg["message"]


@pytest.mark.asyncio
async def test_backpressure_overflow(server_url: str):
    """Blast audio frames to trigger input queue overflow."""
    url = f"{server_url}/v1/stt/ws?call_id=test-bp&sample_rate=16000"
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
    url = f"{server_url}/v1/stt/ws?call_id=test-commit&sample_rate=16000&partial_ms=50"
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
```

**Step 3: Run integration tests**

```bash
STT_BACKEND=mock MOCK_INFER_MS=1 uv run pytest tests/test_integration.py -v
```

**Step 4: Fix any failures and re-run until all pass**

**Step 5: Commit**

```bash
git add tests/conftest.py tests/test_integration.py
git commit -m "feat: integration tests — happy path, invalid params, backpressure, commit flow"
```

---

## Task 14: Full test suite run + final commit

**Step 1: Run all tests**

```bash
STT_BACKEND=mock MOCK_INFER_MS=1 uv run pytest tests/ -v
```

**Step 2: Fix any failures**

**Step 3: Final commit with all fixes**

```bash
git add -A
git commit -m "chore: fix any remaining test issues"
```

---

## Task Dependencies

```
Task 1 (project setup)
  ├── Task 2 (config) ─────────────────────┐
  ├── Task 3 (base) ───────────────────────┤
  │     ├── Task 8 (mock backend + tests)  │
  │     └── Task 9 (registry + qwen3)     │
  ├── Task 4 (schema + tests)             ├── Task 10 (worker) → Task 11 (transport) → Task 12 (app) → Task 13 (integration tests) → Task 14 (full run)
  ├── Task 5 (session + tests)            │
  ├── Task 6 (logging) ───────────────────┤
  └── Task 7 (metrics) ───────────────────┘
```

**Parallelizable groups:**
- Tasks 2, 3, 4, 5, 6, 7 can all run in parallel after Task 1
- Tasks 8, 9 can run in parallel after Task 3
- Tasks 10-14 are sequential
