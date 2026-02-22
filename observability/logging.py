from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import structlog

from config import settings


def setup_logging() -> None:
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with HHmm format
    log_filename = datetime.now().strftime("%H%M") + ".log"
    log_filepath = log_dir / log_filename

    # Configure root logger with file and console handlers
    # Using basicConfig with force=True to reset any existing configuration
    logging.basicConfig(
        level=0,
        format="%(message)s",
        force=True,
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Use stdlib LoggerFactory with empty string to get the root logger
    # which has our handlers configured via basicConfig above
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(""),
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
