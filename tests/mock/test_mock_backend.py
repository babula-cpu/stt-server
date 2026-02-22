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
        self.backend.detect_endpoint()
        self.backend.reset_segment()
        assert self.backend._buf_bytes == 0
        assert self.backend._endpoint_fired is False

    def test_close_is_noop(self):
        self.backend.close()
