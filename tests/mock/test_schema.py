import pytest
from protocol.schema import validate_query_params, build_ready, build_partial, build_final, build_error, ConnParams, ValidationError


class TestValidateQueryParams:
    def test_valid_minimal(self):
        params = validate_query_params({"call_id": "00000000000000007000000000000001", "sample_rate": "16000"})
        assert params.call_id == "00000000000000007000000000000001"
        assert params.sample_rate == 16000
        assert params.codec == "pcm_s16le"
        assert params.channels == 1
        assert params.frame_ms == 20
        assert params.partial_ms == 500
        assert params.language == "auto"
        assert params.hotwords is None

    def test_valid_all_params(self):
        params = validate_query_params({
            "call_id": "00000000000000007000000000000002",
            "sample_rate": "16000",
            "codec": "pcm_s16le",
            "channels": "1",
            "frame_ms": "30",
            "partial_ms": "200",
            "language": "zh",
            "hotwords": "hello%20world%2Cfoo",
        })
        assert params.partial_ms == 200
        assert params.language == "zh"
        assert params.hotwords == ["hello world", "foo"]

    def test_missing_call_id(self):
        with pytest.raises(ValidationError, match="call_id"):
            validate_query_params({"sample_rate": "16000"})

    def test_missing_sample_rate(self):
        with pytest.raises(ValidationError, match="sample_rate"):
            validate_query_params({"call_id": "00000000000000007000000000000099"})

    def test_invalid_sample_rate(self):
        with pytest.raises(ValidationError, match="16000"):
            validate_query_params({"call_id": "00000000000000007000000000000099", "sample_rate": "8000"})

    def test_invalid_codec(self):
        with pytest.raises(ValidationError, match="pcm_s16le"):
            validate_query_params({"call_id": "00000000000000007000000000000099", "sample_rate": "16000", "codec": "opus"})

    def test_invalid_channels(self):
        with pytest.raises(ValidationError, match="1"):
            validate_query_params({"call_id": "00000000000000007000000000000099", "sample_rate": "16000", "channels": "2"})


class TestMessageBuilders:
    def test_build_ready(self):
        params = ConnParams(
            call_id="00000000000000007000000000000001", sample_rate=16000, codec="pcm_s16le",
            channels=1, frame_ms=20, partial_ms=500, language="en", hotwords=None,
        )
        msg = build_ready(params)
        assert msg["type"] == "ready"
        assert msg["call_id"] == "00000000000000007000000000000001"
        assert msg["sample_rate"] == 16000
        assert msg["language"] == "en"
        assert "created_at" in msg

    def test_build_partial(self):
        msg = build_partial("00000000000000007000000000000001", 1, "hello")
        assert msg["type"] == "partial"
        assert msg["segment_seq"] == 1
        assert msg["text"] == "hello"
        assert msg["final"] is False

    def test_build_final(self):
        msg = build_final("00000000000000007000000000000001", 1, "hello world", 85, 120, 2000)
        assert msg["type"] == "final"
        assert msg["final"] is True
        assert msg["first_token_ms"] == 85
        assert msg["latency_ms"] == 120
        assert msg["segment_duration_ms"] == 2000

    def test_build_error(self):
        msg = build_error("00000000000000007000000000000001", 1008, "invalid param")
        assert msg["type"] == "error"
        assert msg["code"] == 1008
        assert msg["message"] == "invalid param"
