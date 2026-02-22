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
