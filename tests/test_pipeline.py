import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def make_wav(path: Path, seconds: float = 12.0, rate: int = 44_100, channels: int = 2):
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    data = np.repeat(tone[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBER", "mock")
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path / "data")
    from app.main import app
    with TestClient(app) as c:
        yield c


def wait_for(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/transcriptions/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


def test_load_audio_resamples_to_16k_mono(tmp_path):
    from app.audio import load_audio
    audio = load_audio(make_wav(tmp_path / "a.wav", seconds=2.0))
    assert audio.ndim == 1
    assert abs(len(audio) - 32_000) < 500
    assert audio.dtype == np.float32


def test_upload_and_transcribe(client, tmp_path):
    wav = make_wav(tmp_path / "speech.wav")
    with wav.open("rb") as f:
        r = client.post("/transcriptions", files={"file": ("speech.wav", f, "audio/wav")})
    assert r.status_code == 202

    job = wait_for(client, r.json()["id"])
    assert job["status"] == "completed"
    segments = job["result"]["segments"]
    assert len(segments) == 3
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == pytest.approx(12.0, abs=0.1)


def test_export_srt(client, tmp_path):
    wav = make_wav(tmp_path / "speech.wav", seconds=6)
    with wav.open("rb") as f:
        job_id = client.post("/transcriptions", files={"file": ("speech.wav", f)}).json()["id"]
    wait_for(client, job_id)
    srt = client.get(f"/transcriptions/{job_id}/export?format=srt").text
    assert "00:00:00,000 --> 00:00:05,000" in srt


def test_rejects_unsupported_extension(client):
    r = client.post("/transcriptions", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_rejects_fake_audio(client):
    r = client.post("/transcriptions", files={"file": ("fake.mp3", b"not really audio")})
    assert r.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/transcriptions/doesnotexist").status_code == 404
