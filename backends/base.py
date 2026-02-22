from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ASRResult:
    text: str
    is_partial: bool
    is_endpoint: bool


class ASRBackend(Protocol):
    """Abstract interface that every ASR backend must implement."""

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
        """Called once per connection with session parameters."""
        ...

    def push_audio(self, pcm_data: bytes) -> None:
        """Feed audio chunk to the backend's internal buffer."""
        ...

    def get_partial(self) -> ASRResult | None:
        """Run partial inference on buffered audio. Returns None if no meaningful result."""
        ...

    def finalize(self) -> ASRResult:
        """Force-finalize current segment, return result, and reset internal buffer."""
        ...

    def detect_endpoint(self) -> bool:
        """Check if the backend detects an utterance boundary."""
        ...

    def reset_segment(self) -> None:
        """Reset internal state for the next segment."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...
