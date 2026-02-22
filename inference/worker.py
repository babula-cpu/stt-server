from __future__ import annotations

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
    out_q_sync: janus.SyncQueue[Any],
    troubleshoot: TroubleshootCollector,
    log: Any,
) -> None:
    """Inference worker — runs in a dedicated thread."""

    should_stop = False

    def _emit(event: dict) -> None:
        nonlocal should_stop
        try:
            out_q_sync.put_nowait(event)
        except queue.Full:
            metrics.q_out_overflow_total.inc()
            troubleshoot.record_overflow("out")
            should_stop = True
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
        while not should_stop:
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
            troubleshoot.record_queue_depth(in_q_sync.qsize(), out_q_sync.qsize())
            metrics.q_in_depth.set(in_q_sync.qsize())
            metrics.q_out_depth.set(out_q_sync.qsize())

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
            out_q_sync.put(STOP, timeout=1.0)
        except (queue.Full, janus.SyncQueueShutDown):
            pass
