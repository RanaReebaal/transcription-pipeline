"""Speech-to-text engines.

The rest of the app only depends on the Transcriber protocol, so swapping
faster-whisper for a hosted API (Deepgram, AssemblyAI, ...) means writing
one new class here.
"""

import logging
import os
from typing import Callable, Iterator, Protocol

import numpy as np

from app.audio import SAMPLE_RATE
from app.schemas import Segment

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


def _add_cuda_dlls():
    """Make pip-installed CUDA libraries (requirements-gpu.txt) visible on Windows.

    On Linux they are found automatically. On Windows CTranslate2 loads
    cuBLAS/cuDNN by name, so their folders need to be on the DLL search path.
    """
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for lib in ("cublas", "cudnn", "cuda_nvrtc"):
            bin_dir = os.path.join(root, lib, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


class Transcriber(Protocol):
    def transcribe(
        self, audio: np.ndarray, on_progress: ProgressCallback | None = None
    ) -> tuple[str, Iterator[Segment]]:
        """Return (detected_language, segments)."""
        ...


class WhisperTranscriber:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        _add_cuda_dlls()
        from faster_whisper import WhisperModel

        model_size = model_size or os.getenv("WHISPER_MODEL", "small")
        self.model_name = model_size
        device = device or os.getenv("WHISPER_DEVICE", "auto")

        # The model can load on a GPU and still fail on the first real call
        # if CUDA libraries (cuBLAS/cuDNN) are missing, so run one tiny
        # transcription up front and fall back to CPU if it fails.
        try:
            if device == "cpu":
                raise RuntimeError("CPU requested")
            self.model = WhisperModel(model_size, device=device, compute_type="float16")
            self._warmup()
            self.device = "gpu"
        except (RuntimeError, ValueError) as exc:
            if device != "cpu":
                log.warning("GPU unavailable (%s), using CPU", exc)
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.device = "cpu"
        log.info("Loaded whisper model '%s' on %s", model_size, self.device)

    def _warmup(self):
        segments, _ = self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                                            language="en")
        list(segments)

    def transcribe(self, audio, on_progress=None):
        total = len(audio) / SAMPLE_RATE
        # vad_filter skips silence, which speeds up long recordings and
        # avoids the model hallucinating text in quiet parts.
        raw_segments, info = self.model.transcribe(
            audio, vad_filter=True, beam_size=5
        )

        def segments() -> Iterator[Segment]:
            # faster-whisper yields segments lazily, so a long file is
            # never held as one big result in memory.
            for i, s in enumerate(raw_segments):
                if on_progress and total:
                    on_progress(min(s.end / total, 1.0))
                yield Segment(
                    id=i,
                    start=round(s.start, 2),
                    end=round(s.end, 2),
                    text=s.text.strip(),
                    confidence=round(float(np.exp(s.avg_logprob)), 3),
                )

        return info.language, segments()


class MockTranscriber:
    """Deterministic output for tests and for running without a model."""

    def transcribe(self, audio, on_progress=None):
        duration = len(audio) / SAMPLE_RATE

        def segments():
            t, i = 0.0, 0
            while t < duration:
                end = min(t + 5.0, duration)
                if on_progress:
                    on_progress(end / duration)
                yield Segment(id=i, start=round(t, 2), end=round(end, 2),
                              text=f"mock segment {i}", confidence=1.0)
                t, i = end, i + 1

        return "en", segments()


def build_transcriber() -> Transcriber:
    if os.getenv("TRANSCRIBER", "whisper") == "mock":
        return MockTranscriber()
    return WhisperTranscriber()
