from __future__ import annotations

import threading
from pathlib import Path

import structlog

from backends.base import ASRResult
from config import settings


def _resolve_model_path() -> str:
    """解析模型路径：本地路径优先，否则视为 HuggingFace 仓库 ID"""
    model = settings.vllm_model

    # 检查是否是本地路径
    # 1. 相对路径: ./xxx, ../xxx
    # 2. 绝对路径: /xxx
    # 3. 存在的目录
    if model.startswith(("./", "../", "/")) or Path(model).is_dir():
        return model

    # 否则视为 HuggingFace 仓库 ID
    return model

logger = structlog.get_logger()

# ── Language mapping (BCP-47 → prompt text) ─────────────────────────────────

_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "vi": "Vietnamese",
    "th": "Thai",
}

# ── Module-level ASR model singleton ─────────────────────────────────────────

_asr_model: object | None = None  # Qwen3ASRModel, typed loosely to avoid import at module level
_asr_lock = threading.Lock()
_infer_lock = threading.Lock()  # Serializes inference calls (vLLM not thread-safe)
_model_loaded: bool = False


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _model_loaded


def preload_model() -> None:
    """预加载模型，在服务器启动时调用。"""
    logger.info("qwen_asr.preload.start")
    _get_asr_model()  # 触发模型加载
    logger.info("qwen_asr.preload.done")


def _get_asr_model() -> object:
    global _asr_model, _model_loaded
    if _asr_model is None:
        with _asr_lock:
            if _asr_model is None:
                from qwen_asr import Qwen3ASRModel

                # Build kwargs from settings (only pass valid vLLM parameters)
                llm_kwargs = {
                    "model": _resolve_model_path(),
                    "gpu_memory_utilization": settings.vllm_gpu_memory,
                    "max_new_tokens": settings.vllm_max_tokens,
                }

                # Only add optional params if explicitly configured
                if settings.vllm_max_inference_batch_size > 0:
                    llm_kwargs["max_inference_batch_size"] = settings.vllm_max_inference_batch_size

                logger.info("qwen_asr.loading.start", model=settings.vllm_model, kwargs=llm_kwargs)
                _asr_model = Qwen3ASRModel.LLM(**llm_kwargs)
                _model_loaded = True
                logger.info("qwen_asr.model.loaded", model=settings.vllm_model)
    return _asr_model


class Qwen3Backend:
    """Qwen3-ASR via qwen-asr streaming API."""

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._language = language
        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._endpoint_fired = False

        asr = _get_asr_model()
        lang_name = None if language == "auto" else _LANG_MAP.get(language, language)
        self._state = asr.init_streaming_state(
            language=lang_name,
            chunk_size_sec=settings.streaming_chunk_size_sec,
            unfixed_chunk_num=settings.streaming_unfixed_chunk_num,
            unfixed_token_num=settings.streaming_unfixed_token_num,
        )

    def push_audio(self, pcm_data: bytes) -> None:
        import numpy as np

        n_samples = len(pcm_data) // 2
        if n_samples == 0:
            return

        # Single numpy pass: parse int16, compute RMS, convert to float32
        samples_i16 = np.frombuffer(pcm_data[: n_samples * 2], dtype=np.int16)
        rms = np.sqrt(np.mean(samples_i16.astype(np.int64) ** 2))

        chunk_ms = n_samples / self._sample_rate * 1000
        if rms >= settings.vad_threshold:
            self._in_speech = True
            self._silence_ms_accum = 0.0
        elif self._in_speech:
            self._silence_ms_accum += chunk_ms

        # Reuse parsed int16 for float32 conversion
        audio = samples_i16.astype(np.float32) / 32768.0
        asr = _get_asr_model()
        with _infer_lock:
            asr.streaming_transcribe(audio, self._state)

    def detect_endpoint(self) -> bool:
        if (
            self._in_speech
            and not self._endpoint_fired
            and self._silence_ms_accum >= settings.vad_silence_ms
        ):
            self._endpoint_fired = True
            return True
        return False

    def get_partial(self) -> ASRResult | None:
        if not self._state.text:
            return None
        return ASRResult(text=self._state.text, is_partial=True, is_endpoint=False)

    def finalize(self) -> ASRResult:
        logger.info("finalize.called")
        asr = _get_asr_model()
        with _infer_lock:
            asr.finish_streaming_transcribe(self._state)
        text = self._state.text or ""
        logger.info("finalize.done", text=text)
        return ASRResult(text=text, is_partial=False, is_endpoint=True)

    def reset_segment(self) -> None:
        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._endpoint_fired = False

        asr = _get_asr_model()
        lang_name = None if self._language == "auto" else _LANG_MAP.get(self._language, self._language)
        self._state = asr.init_streaming_state(
            language=lang_name,
            chunk_size_sec=settings.streaming_chunk_size_sec,
            unfixed_chunk_num=settings.streaming_unfixed_chunk_num,
            unfixed_token_num=settings.streaming_unfixed_token_num,
        )

    def close(self) -> None:
        pass  # model is shared singleton, not per-connection
