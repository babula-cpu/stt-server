import time
from session.state import SessionState


class TestSessionState:
    def test_initial_state(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        assert s.segment_seq == 1
        assert s.segment_bytes == 0
        assert s.first_audio_ts is None
        assert s.first_partial_ts is None

    def test_record_audio(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        s.record_audio(3200)
        assert s.segment_bytes == 3200
        assert s.first_audio_ts is not None
        first = s.first_audio_ts
        s.record_audio(3200)
        assert s.segment_bytes == 6400
        assert s.first_audio_ts == first  # unchanged

    def test_record_partial(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        s.record_partial()
        assert s.first_partial_ts is not None
        assert s.last_partial_ts > 0

    def test_should_send_partial(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        s.partial_ms = 500
        # First partial always allowed
        assert s.should_send_partial() is True
        s.record_partial()
        # Immediately after, throttled
        assert s.should_send_partial() is False

    def test_increment_segment(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        s.record_audio(3200)
        s.record_partial()
        s.increment_segment()
        assert s.segment_seq == 2
        assert s.segment_bytes == 0
        assert s.first_audio_ts is None
        assert s.first_partial_ts is None

    def test_compute_timing(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        s.sample_rate = 16000
        s.record_audio(32000)  # 1 second of audio
        # Simulate some passage of time by setting timestamps directly
        s.first_audio_ts = time.monotonic() - 0.5
        s.first_partial_ts = s.first_audio_ts + 0.085
        timing = s.compute_timing()
        assert timing["segment_duration_ms"] == 1000
        assert 84 <= timing["first_token_ms"] <= 86  # Allow for rounding
        assert "latency_ms" in timing

    def test_segment_duration_from_bytes(self):
        s = SessionState(conn_id="abc123", call_id="00000000000000007000000000000001")
        s.sample_rate = 16000
        s.record_audio(64000)  # 2 seconds
        timing = s.compute_timing()
        assert timing["segment_duration_ms"] == 2000
