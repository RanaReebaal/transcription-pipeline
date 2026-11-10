"""Command-line version: transcribe one file without running the server.

    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --format srt -o audio.srt
"""

import argparse
import sys
from pathlib import Path

from app import formats
from app.audio import InvalidAudioError, load_audio, probe
from app.schemas import Transcript
from app.transcriber import build_transcriber


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument("file", type=Path)
    parser.add_argument("--format", choices=["json", "srt", "vtt", "txt"], default="json")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    try:
        info = probe(args.file)
        audio = load_audio(args.file)
    except InvalidAudioError as exc:
        sys.exit(f"error: {exc}")

    print(f"{args.file.name}: {info.codec}, {info.sample_rate} Hz, "
          f"{info.channels} ch, {info.duration}s", file=sys.stderr)

    def progress(p):
        print(f"\r{p:6.1%}", end="", file=sys.stderr, flush=True)

    language, segments = build_transcriber().transcribe(audio, progress)
    segments = list(segments)
    print(file=sys.stderr)

    transcript = Transcript(
        language=language,
        duration=round(len(audio) / 16_000, 2),
        text=" ".join(s.text for s in segments),
        segments=segments,
    )
    out = {
        "json": lambda t: t.model_dump_json(indent=2),
        "srt": formats.to_srt,
        "vtt": formats.to_vtt,
        "txt": formats.to_text,
    }[args.format](transcript)

    if args.output:
        args.output.write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()
