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
q_in_frames_dropped_total = Counter("stt_q_in_frames_dropped_total", "Input queue frames dropped (oldest discarded)")
q_out_overflow_total = Counter("stt_q_out_overflow_total", "Output queue overflow count")
partials_dropped_backpressure = Counter("stt_partials_dropped_backpressure", "Partial results dropped due to output queue backpressure")

# Inference
asr_infer_ms = Histogram("stt_asr_infer_ms", "ASR inference duration in ms", buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000])
ttfb_ms = Histogram("stt_ttfb_ms", "Time to first token in ms", buckets=[25, 50, 100, 250, 500, 1000, 2500])
final_latency_ms = Histogram("stt_final_latency_ms", "Final result latency in ms", buckets=[50, 100, 250, 500, 1000, 2500, 5000])

# Segment
finalize_total = Counter("stt_finalize_total", "Total finalizations", ["reason"])
