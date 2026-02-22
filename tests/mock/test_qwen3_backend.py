from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

from backends.base import ASRResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pcm_silence(n_samples: int) -> bytes:
    """PCM16LE silence (all zeros)."""
    return b"\x00\x00" * n_samples


def _make_pcm_tone(n_samples: int, amplitude: int = 10000) -> bytes:
    """PCM16LE constant-amplitude samples (square wave)."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _make_mock_asr():
    """Create a mock qwen-asr model that returns fresh state objects."""
    mock_asr = MagicMock()
    mock_asr.init_streaming_state.side_effect = (
        lambda **kwargs: MagicMock(text="", language="")
    )
    return mock_asr


def _make_backend(mock_asr=None, language="auto"):
    """Create and configure a Qwen3Backend with _get_asr_model mocked out."""
    if mock_asr is None:
        mock_asr = _make_mock_asr()
    with patch("backends.qwen3._get_asr_model", return_value=mock_asr):
        from backends.qwen3 import Qwen3Backend

        b = Qwen3Backend()
        b.configure(sample_rate=16000, language=language)
    return b


# ── VAD tests ────────────────────────────────────────────────────────────────


class TestVAD:
    def setup_method(self):
        self.mock_asr = _make_mock_asr()
        self.patcher = patch(
            "backends.qwen3._get_asr_model", return_value=self.mock_asr
        )
        self.patcher.start()
        self.backend = _make_backend(self.mock_asr)

    def teardown_method(self):
        self.patcher.stop()

    def test_silence_does_not_trigger_speech(self):
        self.backend.push_audio(_make_pcm_silence(1600))
        assert self.backend._in_speech is False

    def test_loud_audio_triggers_speech(self):
        self.backend.push_audio(_make_pcm_tone(1600, amplitude=10000))
        assert self.backend._in_speech is True

    def test_endpoint_after_sustained_silence(self):
        """Speech followed by enough silence triggers endpoint."""
        # Speech chunk
        self.backend.push_audio(_make_pcm_tone(1600, amplitude=10000))
        assert self.backend._in_speech is True

        # 800ms of silence at 16kHz (12800 samples) > default 700ms threshold
        self.backend.push_audio(_make_pcm_silence(12800))
        assert self.backend.detect_endpoint() is True

    def test_endpoint_fires_once(self):
        self.backend.push_audio(_make_pcm_tone(1600, amplitude=10000))
        self.backend.push_audio(_make_pcm_silence(12800))
        assert self.backend.detect_endpoint() is True
        assert self.backend.detect_endpoint() is False

    def test_speech_resets_silence_accumulator(self):
        """Intermittent speech should reset silence accumulation."""
        # Speech
        self.backend.push_audio(_make_pcm_tone(1600, amplitude=10000))
        # Short silence (400ms = 6400 samples, < 700ms threshold)
        self.backend.push_audio(_make_pcm_silence(6400))
        assert self.backend.detect_endpoint() is False
        # More speech resets accumulator
        self.backend.push_audio(_make_pcm_tone(1600, amplitude=10000))
        assert self.backend._silence_ms_accum == 0.0
        # Another short silence — still no endpoint
        self.backend.push_audio(_make_pcm_silence(6400))
        assert self.backend.detect_endpoint() is False

    def test_no_endpoint_without_speech(self):
        """Silence alone should never fire an endpoint."""
        self.backend.push_audio(_make_pcm_silence(32000))
        assert self.backend.detect_endpoint() is False


# ── Reset tests ──────────────────────────────────────────────────────────────


class TestResetSegment:
    def test_reset_clears_all_state(self):
        mock_asr = _make_mock_asr()
        patcher = patch("backends.qwen3._get_asr_model", return_value=mock_asr)
        patcher.start()
        try:
            backend = _make_backend(mock_asr)
            backend.push_audio(_make_pcm_tone(1600, amplitude=10000))
            backend.push_audio(_make_pcm_silence(12800))
            backend.detect_endpoint()

            old_state = backend._state
            backend.reset_segment()

            assert backend._state is not old_state
            assert backend._in_speech is False
            assert backend._silence_ms_accum == 0.0
            assert backend._endpoint_fired is False
        finally:
            patcher.stop()


# ── Transcribe tests ─────────────────────────────────────────────────────────


class TestTranscribe:
    def setup_method(self):
        self.mock_asr = _make_mock_asr()
        self.patcher = patch(
            "backends.qwen3._get_asr_model", return_value=self.mock_asr
        )
        self.patcher.start()

    def teardown_method(self):
        self.patcher.stop()

    def test_get_partial_empty_state_returns_none(self):
        backend = _make_backend(self.mock_asr)
        assert backend.get_partial() is None

    def test_get_partial_with_text(self):
        backend = _make_backend(self.mock_asr)
        backend._state.text = "hello world"
        result = backend.get_partial()
        assert result is not None
        assert result.text == "hello world"
        assert result.is_partial is True
        assert result.is_endpoint is False

    def test_finalize_calls_finish_streaming(self):
        backend = _make_backend(self.mock_asr)
        backend._state.text = "final text"
        result = backend.finalize()
        self.mock_asr.finish_streaming_transcribe.assert_called_once_with(
            backend._state
        )
        assert result.text == "final text"
        assert result.is_partial is False
        assert result.is_endpoint is True

    def test_finalize_empty_state(self):
        backend = _make_backend(self.mock_asr)
        result = backend.finalize()
        assert result.text == ""
        assert result.is_partial is False
        assert result.is_endpoint is True

    def test_push_audio_calls_streaming_transcribe(self):
        import numpy as np

        backend = _make_backend(self.mock_asr)
        pcm = _make_pcm_tone(1600, amplitude=5000)
        backend.push_audio(pcm)

        self.mock_asr.streaming_transcribe.assert_called_once()
        call_args = self.mock_asr.streaming_transcribe.call_args
        audio_arg = call_args[0][0]
        state_arg = call_args[0][1]

        # Verify audio is float32 normalized
        assert audio_arg.dtype == np.float32
        expected = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_equal(audio_arg, expected)
        # Verify it was called with the backend's state
        assert state_arg is backend._state

    def test_error_propagation(self):
        backend = _make_backend(self.mock_asr)
        self.mock_asr.streaming_transcribe.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(RuntimeError, match="GPU OOM"):
            backend.push_audio(_make_pcm_tone(1600))


# ── Configure & Close tests ──────────────────────────────────────────────────


class TestConfigureAndClose:
    def test_configure_initializes_state(self):
        mock_asr = _make_mock_asr()
        with patch("backends.qwen3._get_asr_model", return_value=mock_asr):
            backend = _make_backend(mock_asr)
        assert backend._sample_rate == 16000
        assert backend._language == "auto"
        assert backend._state is not None
        assert backend._in_speech is False
        assert backend._endpoint_fired is False

    def test_configure_language_mapping(self):
        mock_asr = _make_mock_asr()
        with patch("backends.qwen3._get_asr_model", return_value=mock_asr):
            backend = _make_backend(mock_asr, language="zh")
        # init_streaming_state should have been called with language="Chinese"
        mock_asr.init_streaming_state.assert_called_with(
            language="Chinese",
            chunk_size_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )

    def test_configure_auto_language(self):
        mock_asr = _make_mock_asr()
        with patch("backends.qwen3._get_asr_model", return_value=mock_asr):
            backend = _make_backend(mock_asr, language="auto")
        mock_asr.init_streaming_state.assert_called_with(
            language=None,
            chunk_size_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )

    def test_close_is_noop(self):
        backend = _make_backend()
        backend.close()  # should not raise

    def test_close_idempotent(self):
        backend = _make_backend()
        backend.close()
        backend.close()  # should not raise
