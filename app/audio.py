"""Audio validation and normalization.

Every input is decoded with PyAV (FFmpeg bindings) and resampled to
16 kHz mono float32, which is what Whisper expects. This means WAV, MP3,
M4A, OGG, FLAC and even video files with an audio track all go through
one code path.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


class InvalidAudioError(ValueError):
    pass


@dataclass
class AudioInfo:
    duration: float
    codec: str
    sample_rate: int
    channels: int


def probe(path: Path) -> AudioInfo:
    """Inspect the actual file contents instead of trusting the extension."""
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InvalidAudioError(f"Unsupported file type: {path.suffix or 'none'}")

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise InvalidAudioError("File has no audio stream")
            stream = container.streams.audio[0]
            duration = 0.0
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = container.duration / av.time_base
            return AudioInfo(
                duration=round(duration, 2),
                codec=stream.codec_context.name,
                sample_rate=stream.codec_context.sample_rate,
                channels=stream.codec_context.channels,
            )
    except av.error.FFmpegError as exc:
        log.info("Decode failed for %s: %s", path.name, exc)
        raise InvalidAudioError("Could not decode audio file") from exc


def load_audio(path: Path) -> np.ndarray:
    """Decode any supported file to a 16 kHz mono float32 array."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray().reshape(-1))
    except av.error.FFmpegError as exc:
        log.info("Decode failed for %s: %s", path.name, exc)
        raise InvalidAudioError("Could not decode audio file") from exc

    if not chunks:
        raise InvalidAudioError("Audio stream is empty")
    return np.concatenate(chunks).astype(np.float32) / 32768.0
