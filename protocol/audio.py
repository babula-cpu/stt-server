"""Audio processing utilities."""

import struct
from pathlib import Path


class WavDecodeError(Exception):
    """Raised when WAV decoding fails."""
    pass


def decode_wav_to_pcm16le(wav_data: bytes) -> bytes:
    """Decode WAV audio to PCM16LE mono 16kHz.

    Supports:
    - PCM: fmt_tag=1, 16-bit or 32-bit
    - IEEE Float: fmt_tag=3, 32-bit (float) or 64-bit (double)

    Processing:
    - Mixdown multi-channel to mono
    - Resample to 16kHz via linear interpolation
    - Convert to PCM16LE

    Args:
        wav_data: Raw WAV file content

    Returns:
        PCM16LE mono 16kHz audio bytes

    Raises:
        WavDecodeError: If WAV format is invalid or unsupported
    """
    # Parse RIFF header
    if len(wav_data) < 44:
        raise WavDecodeError(f"WAV file too small: {len(wav_data)} bytes")

    # Verify RIFF header
    if wav_data[:4] != b"RIFF":
        raise WavDecodeError("Invalid WAV: missing RIFF header")
    if wav_data[8:12] != b"WAVE":
        raise WavDecodeError("Invalid WAV: missing WAVE format")

    src_rate = 0
    src_channels = 0
    src_bits = 0
    fmt_tag = 0
    data_bytes = b""

    pos = 12
    file_size = len(wav_data)

    while pos < file_size:
        if pos + 8 > file_size:
            break

        chunk_id = wav_data[pos : pos + 4]
        chunk_size = struct.unpack("<I", wav_data[pos + 4 : pos + 8])[0]

        if chunk_id == b"fmt ":
            fmt_data = wav_data[pos + 8 : pos + 8 + chunk_size]
            fmt_tag = struct.unpack_from("<H", fmt_data, 0)[0]
            src_channels = struct.unpack_from("<H", fmt_data, 2)[0]
            src_rate = struct.unpack_from("<I", fmt_data, 4)[0]
            src_bits = struct.unpack_from("<H", fmt_data, 14)[0]
        elif chunk_id == b"data":
            data_bytes = wav_data[pos + 8 : pos + 8 + chunk_size]
            break

        pos += 8 + chunk_size

    if not data_bytes:
        raise WavDecodeError("WAV file missing data chunk")

    if fmt_tag not in [1, 3]:
        raise WavDecodeError(f"Unsupported WAV format tag: {fmt_tag}. Only PCM (1) and IEEE Float (3) are supported.")

    # Decode samples to float list
    if fmt_tag == 3:  # IEEE float
        n_samples = len(data_bytes) // (src_bits // 8)
        if src_bits == 32:
            fmt_char = "f"
        elif src_bits == 64:
            fmt_char = "d"
        else:
            raise WavDecodeError(f"Unsupported float bit depth: {src_bits}")
        samples = list(struct.unpack(f"<{n_samples}{fmt_char}", data_bytes))
    elif fmt_tag == 1:  # PCM
        if src_bits == 16:
            n_samples = len(data_bytes) // 2
            samples = [s / 32768.0 for s in struct.unpack(f"<{n_samples}h", data_bytes)]
        elif src_bits == 32:
            n_samples = len(data_bytes) // 4
            samples = [s / 2147483648.0 for s in struct.unpack(f"<{n_samples}i", data_bytes)]
        else:
            raise WavDecodeError(f"Unsupported PCM bit depth: {src_bits}. Only 16-bit and 32-bit are supported.")

    # Mono mixdown if needed
    if src_channels > 1:
        mono = []
        for i in range(0, len(samples), src_channels):
            mono.append(sum(samples[i : i + src_channels]) / src_channels)
        samples = mono

    # Resample to 16kHz via linear interpolation
    target_rate = 16000
    if src_rate != target_rate:
        ratio = src_rate / target_rate
        n_target = int(len(samples) / ratio)
        resampled = []
        for i in range(n_target):
            src_pos = i * ratio
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(samples):
                val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            else:
                val = samples[idx]
            resampled.append(val)
        samples = resampled

    # Convert to int16 PCM
    pcm_samples = []
    for s in samples:
        clipped = max(-1.0, min(1.0, s))
        pcm_samples.append(int(clipped * 32767))

    # Pack as little-endian
    return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)


def load_wav_as_pcm16le_16khz(wav_path: Path) -> bytes:
    """Convenience function to load WAV file from path.

    Args:
        wav_path: Path to WAV file

    Returns:
        PCM16LE mono 16kHz audio bytes
    """
    return decode_wav_to_pcm16le(wav_path.read_bytes())
