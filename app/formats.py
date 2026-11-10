"""Export a transcript for downstream use."""

from app.schemas import Transcript


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def to_srt(t: Transcript) -> str:
    blocks = [
        f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(t.segments, 1)
    ]
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for s in t.segments:
        lines += [f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}", s.text, ""]
    return "\n".join(lines)


def to_text(t: Transcript) -> str:
    return t.text + "\n"
