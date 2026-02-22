# STT WebSocket Interface

Speech-to-Text streaming API using WebSocket for real-time transcription.

## Endpoint

```
WebSocket /v1/stt/ws
```

## Query Parameters

| Parameter     | Type   | Required | Default     | Description                              |
|---------------|--------|----------|-------------|------------------------------------------|
| `call_id`     | string | Yes      | -           | Call identifier for tracing/logs         |
| `sample_rate` | int    | Yes      | -           | Audio sample rate (must be 16000)        |
| `codec`       | string | No       | `pcm_s16le` | Audio codec (only `pcm_s16le` supported) |
| `channels`    | int    | No       | 1           | Number of channels (only 1 supported)    |
| `frame_ms`    | int    | No       | 20          | Client frame duration hint in ms         |
| `partial_ms`  | int    | No       | 500         | Partial result throttle interval in ms   |
| `language`    | string | No       | `auto`      | BCP-47 language tag (e.g., `en`, `zh`, `ja`). `auto` lets the backend auto-detect. |
| `hotwords`    | string | No       | -           | URL-encoded comma-separated hotwords for recognition bias |

### Hotwords Parameter

The `hotwords` parameter improves recognition accuracy for domain-specific vocabulary. Pass a comma-separated list of words/phrases that should be prioritized during recognition.

**Encoding:** The value must be URL-encoded (percent-encoded). Use `encodeURIComponent()` in JavaScript or `urllib.parse.quote()` in Python.

**Example:**
```
# Original hotwords: caesar salad,grilled salmon,sparkling water
# URL-encoded: caesar%20salad%2Cgrilled%20salmon%2Csparkling%20water

ws://localhost:8000/v1/stt/ws?call_id=test&sample_rate=16000&language=en&hotwords=caesar%20salad%2Cgrilled%20salmon%2Csparkling%20water
```

## Connection Flow

```
Client                                Server
  |                                      |
  |  ---- WebSocket Connect ---------->  |
  |  <--- ready (JSON) ----------------  |
  |                                      |
  |  ---- Binary audio frames -------->  |
  |  <--- partial (seq=1) -------------  |  (throttled by partial_ms)
  |  ---- Binary audio frames -------->  |
  |  <--- partial (seq=1) -------------  |
  |  ---- Binary audio frames -------->  |
  |  <--- final   (seq=1) -------------  |  (on endpoint detection, seq++)
  |                                      |
  |  ---- Binary audio frames -------->  |
  |  <--- partial (seq=2) -------------  |
  |  ---- commit (JSON) -------------->  |  (optional: client-side endpoint)
  |  <--- final   (seq=2) -------------  |  (immediate response, seq++)
  |                                      |
  |  ---- WebSocket Close ------------>  |
  |                                      |
```

## Segment Sequence (`segment_seq`)

Each `partial` and `final` message carries a `segment_seq` field for observability.

| Property       | Value                                                        |
|----------------|--------------------------------------------------------------|
| Start value    | 1                                                            |
| Increment      | Only on sending a `final` message (i.e. utterance committed) |
| Scope          | Per `call_id`                                                |

- All `partial` messages for an in-progress utterance share the same `segment_seq` as the eventual `final`.
- Rejected segments (empty/noise) do not produce a `final` and do not increment `segment_seq`.
- Use `call_id + segment_seq` to uniquely locate a completed utterance segment across logs and metrics.

**Example timeline:**

```
partial  segment_seq=1  text="I would"
partial  segment_seq=1  text="I would like the"
final    segment_seq=1  text="I would like the caesar salad"   ← seq increments after this
partial  segment_seq=2  text="and"
partial  segment_seq=2  text="and a sparkling"
final    segment_seq=2  text="and a sparkling water please"    ← seq increments after this
```

## Messages

All server-to-client messages include a `created_at` field: UTC Unix timestamp (seconds, with fractional milliseconds) indicating when the event was created.

### Server -> Client

#### ready

Sent immediately after connection is accepted.

```json
{
  "type": "ready",
  "call_id": "test-001",
  "created_at": 1739181600.123,
  "sample_rate": 16000,
  "codec": "pcm_s16le",
  "channels": 1,
  "frame_ms": 20,
  "partial_ms": 500,
  "language": "en"
}
```

#### partial

Sent when intermediate recognition result is available (throttled by `partial_ms`).

```json
{
  "type": "partial",
  "call_id": "test-001",
  "created_at": 1739181600.456,
  "segment_seq": 1,
  "text": "I would like the",
  "final": false
}
```

#### final

Sent when endpoint is detected (speaker finished an utterance).

```json
{
  "type": "final",
  "call_id": "test-001",
  "created_at": 1739181602.789,
  "segment_seq": 1,
  "text": "I would like the caesar salad",
  "final": true,
  "first_token_ms": 85,
  "latency_ms": 120,
  "segment_duration_ms": 2000
}
```

Fields in final response:

| Field               | Type | Description                                      |
|---------------------|------|--------------------------------------------------|
| `segment_seq`       | int  | Segment sequence number (1-based, see below)     |
| `first_token_ms`    | int  | TTFB: Time to first token in streaming mode (ms) |
| `latency_ms`        | int  | Total inference latency from submit to result    |
| `segment_duration_ms` | int | Duration of the audio segment (ms)              |

#### error

Sent when an error occurs. Connection is closed after this message.

```json
{
  "type": "error",
  "call_id": "test-001",
  "created_at": 1739181600.001,
  "code": 1008,
  "message": "Invalid sample_rate: 8000. Must be 16000."
}
```

### Client -> Server

#### Binary Audio Frames

Send raw PCM16LE audio bytes as binary WebSocket frames.

Audio format requirements:
- Encoding: PCM signed 16-bit little-endian
- Sample rate: 16000 Hz
- Channels: 1 (mono)

Recommended chunk size: 3200 bytes (100ms at 16kHz)

#### commit

Client sends to request immediate finalization of current utterance.

```json
{
  "type": "commit"
}
```

Server will respond with a `final` message containing transcription accumulated so far,
then reset for the next utterance. This allows client-side VAD to control endpoint
detection instead of waiting for server-side VAD, reducing latency.

## Error Codes

| Code | Description                                |
|------|--------------------------------------------|
| 1000 | Normal closure                             |
| 1008 | Policy violation (invalid parameters)      |
| 1011 | Internal server error                      |


## Notes

- Stream supports multiple utterances; after each `final` result, the stream resets for the next utterance
- Add silence padding (~1s) at end of audio to help endpoint detection
- The `partial_ms` parameter controls how often partial results are sent (minimum interval)
- Empty or whitespace-only results are not sent
- Hotwords improve recognition for domain-specific vocabulary; always URL-encode the parameter value

---

# STT HTTP Interface (Offline Transcription)

HTTP POST endpoint for offline (non-streaming) audio transcription.

## Motivation

The HTTP interface provides a simpler integration path for:
- Batch transcription jobs
- Simple request-response workflows
- Environments where WebSocket is not available
- Quick prototyping and testing

**Non-goals:**
- Not for real-time streaming (use WebSocket `/v1/stt/ws` instead)
- Not for very long audio files (max 100MB)
- Not for non-WAV audio formats

## Endpoint

```
POST /v1/stt/transcribe
```

## Request

**Content-Type:** `multipart/form-data`

| Parameter  | Location    | Type       | Required | Default | Description                              |
|------------|-------------|------------|----------|---------|------------------------------------------|
| `file`     | form-data   | file       | Yes      | -       | WAV audio file                           |
| `language` | query       | string     | No       | `auto`  | Language code (e.g., `zh`, `en`, `auto`) |

### Audio Requirements

The `decode_wav_to_pcm16le()` function handles WAV decoding:

| Format        | WAV fmt_tag | Bit Depth                    |
|---------------|-------------|------------------------------|
| PCM           | 1           | 16-bit, 32-bit              |
| IEEE Float    | 3           | 32-bit (float), 64-bit (double) |

- Input: Any sample rate, any number of channels
- Processing: Mixdown to mono → resample to 16kHz → convert to PCM16LE
- Non-WAV files or unsupported formats return 422

## Processing Flow

```
1. Read file → Size check (max_upload_bytes)
2. WAV decode to PCM16LE 16kHz mono
3. Execute sync inference in asyncio.to_thread:
   - create_backend() → configure(16kHz, language)
   - push_audio(pcm) → finalize() → close()
4. Return result
```

**Note:** Model is not reloaded per request. The backend singleton is reused.

## Success Response

**Status:** `200 OK`

```json
{
  "text": "转写文本内容",
  "duration_ms": 3200.0,
  "latency_ms": 450.3
}
```

| Field        | Type   | Description                                                                      |
|--------------|--------|----------------------------------------------------------------------------------|
| `text`       | string | Transcription result text                                                        |
| `duration_ms`| float  | Audio duration (ms): `len(pcm) / 2 / 16000 * 1000`                             |
| `latency_ms` | float  | Inference time (ms)                                                             |

## Error Responses

| Status | Condition                                      | Body                                            |
|--------|-----------------------------------------------|------------------------------------------------|
| 413    | File exceeds `max_upload_bytes`               | `{"detail": "File too large (N bytes). Max: M"}` |
| 422    | WAV decode failure (non-WAV, unsupported)    | `{"detail": "<specific error message>"}`       |
| 500    | Inference error                               | `{"detail": "Transcription failed"}`            |

## cURL Example

```bash
curl -X POST http://localhost:8000/v1/stt/transcribe \
  -F "file=@audio.wav" \
  -G -d "language=zh"
```

## Observability

- Request logs include: file size, duration_ms, latency_ms
- Backend metrics track inference time
- Error responses include detailed messages for debugging
