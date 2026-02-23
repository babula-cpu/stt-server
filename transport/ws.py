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

        # Create queues — both janus for clean thread/async bridging
        in_janus: janus.Queue[Any] = janus.Queue(maxsize=settings.in_queue_size)
        out_janus: janus.Queue[Any] = janus.Queue(maxsize=settings.out_queue_size)

        # Send ready
        ready_msg = build_ready(conn_params)
        await websocket.send_json(ready_msg)
        troubleshoot.record_event("ws.ready_sent")
        metrics.out_events_total.labels(type="ready").inc()
        log.info("ws.ready_sent")

        # Launch pumps + worker
        recv_task = asyncio.create_task(
            _recv_pump(websocket, in_janus.async_q, out_janus.async_q, session, log, troubleshoot)
        )
        send_task = asyncio.create_task(
            _send_pump(websocket, out_janus.async_q, log, troubleshoot)
        )
        worker_task = asyncio.ensure_future(
            asyncio.to_thread(
                run_worker,
                backend=backend,
                session=session,
                in_q_sync=in_janus.sync_q,
                out_q_sync=out_janus.sync_q,
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
            out_janus.close()
            await out_janus.wait_closed()

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
    out_q: janus.AsyncQueue[Any],
    session: SessionState,
    log: Any,
    troubleshoot: TroubleshootCollector,
) -> None:
    overflow_count = 0
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
                    # Drop oldest frame and enqueue the new one
                    try:
                        in_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        in_q.put_nowait(pcm_data)
                    except asyncio.QueueFull:
                        pass
                    metrics.q_in_overflow_total.inc()
                    metrics.q_in_frames_dropped_total.inc()
                    troubleshoot.record_overflow("in")
                    overflow_count += 1
                    if overflow_count % 50 == 1:
                        log.warning("queue.in.drop_oldest", q_size=in_q.qsize(), total_dropped=overflow_count)
                    if overflow_count >= settings.in_queue_max_drops:
                        troubleshoot.record_event("queue.in.overflow_limit")
                        log.error("queue.in.overflow_limit", total_dropped=overflow_count)
                        err = build_error(session.call_id, 1008, "audio queue overflow")
                        try:
                            await asyncio.wait_for(out_q.put(err), timeout=2.0)
                        except (asyncio.TimeoutError, asyncio.QueueFull):
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
    out_q: janus.AsyncQueue[Any],
    log: Any,
    troubleshoot: TroubleshootCollector,
) -> None:
    try:
        while True:
            try:
                event = await asyncio.wait_for(out_q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except janus.AsyncQueueShutDown:
                break

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
