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
