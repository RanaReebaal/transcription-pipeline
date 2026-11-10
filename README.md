# Transcription Pipeline

A small service that takes an audio file, transcribes it with an open-source
speech-to-text model, and returns the text with timestamps per segment.
It can also export the result as SRT, VTT or plain text for downstream use.

Built with **FastAPI** and **faster-whisper** (Whisper running on CTranslate2).

## Quick start

Requires Python 3.10+. No system FFmpeg needed (PyAV bundles it). Runs on CPU
out of the box; uses an NVIDIA GPU automatically if CUDA 12 and cuDNN are installed.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**Run the API**

```bash
uvicorn app.main:app --port 8080
```

Then open http://localhost:8080/docs for interactive docs, or:

```bash
curl -F "file=@samples/sample.wav" http://localhost:8080/transcriptions
# -> {"id": "3f2a...", "status": "queued", ...}

curl http://localhost:8080/transcriptions/3f2a...
# -> status, progress, and the transcript once completed

curl "http://localhost:8080/transcriptions/3f2a.../export?format=srt"
```

**Or use the command line**

```bash
python transcribe.py samples/sample.wav
python transcribe.py samples/sample.wav --format srt -o sample.srt
```

**Run the tests** (these use a mock transcriber, so no model download is needed)

```bash
pytest
```

The first real run downloads the Whisper model (~470 MB for `small`).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | `cuda`, `cpu` or `auto`; falls back to CPU if CUDA isn't available |
| `TRANSCRIBER` | `whisper` | `mock` returns fake segments (for tests / demos without a model) |
| `WORKERS` | `2` | Number of transcription jobs that run in parallel |
| `MAX_UPLOAD_MB` | `200` | Upload size limit |
| `DATA_DIR` | `data` | Where uploads and job files are stored |

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/transcriptions` | Upload a file (multipart `file`). Returns `202` with a job. |
| `GET` | `/transcriptions/{id}` | Job status, progress (0 to 1) and result |
| `GET` | `/transcriptions/{id}/export?format=srt\|vtt\|txt` | Transcript in a downstream format |
| `GET` | `/health` | Health check |

Example result:

```json
{
  "id": "3f2a...",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "language": "en",
    "duration": 41.3,
    "text": "Full transcript ...",
    "segments": [
      {"id": 0, "start": 0.0, "end": 4.52, "text": "First sentence.", "confidence": 0.91}
    ]
  }
}
```

## Project layout

```
app/
  main.py         HTTP endpoints, upload validation
  audio.py        Probing and decoding any format to 16 kHz mono
  transcriber.py  Whisper engine + mock engine behind one interface
  jobs.py         Job storage on disk, worker pool, retries
  formats.py      SRT / VTT / text export
  schemas.py      Pydantic models
transcribe.py     Command-line entry point
tests/            API and audio tests
```

## Design decisions

**Asynchronous jobs instead of a blocking request.**
Transcribing an hour of audio can take minutes. The upload endpoint saves the
file, returns `202` with a job id straight away, and a background worker does
the work. The client polls for status and progress. This keeps HTTP requests
short and makes long files a normal case rather than a special one.

**One normalization step for every format.**
Instead of handling each format separately, every file is decoded with PyAV
(FFmpeg bindings) and resampled to 16 kHz, mono, float32, which is what Whisper
expects. WAV, MP3, M4A, OGG, FLAC and video files with audio all go through the
same code. PyAV ships its own FFmpeg libraries, so there's nothing extra to install.

**Check the content, not the extension.**
The extension is checked first as a cheap filter, then the file is actually
opened to confirm it has a decodable audio stream. A renamed text file or a
corrupt upload is rejected with a clear `400` before it reaches the queue.

**Long audio.**
- Processed in the background, never inside the request.
- faster-whisper works through the audio in 30-second windows and yields
  segments one at a time, so memory doesn't grow with the length of the transcript.
- Voice activity detection (`vad_filter=True`) skips silence. That makes long
  recordings faster and stops the model from inventing text in quiet sections.
- Progress is updated after each segment, so the client can show how far along it is.
- Uploads are streamed to disk in 1 MB chunks with a size limit, so a large file
  is never held in memory.

**Swappable speech-to-text engine.**
The pipeline only depends on a small `Transcriber` interface. Moving to a hosted
API (Deepgram, AssemblyAI, OpenAI) would mean adding one class. The same
interface lets the tests run with a mock engine, so they're fast and don't need
a GPU or a model download.

**Model loaded once.**
The Whisper model loads at startup and is shared by all jobs. Loading it per
request would add seconds to every call. It uses the GPU with float16 when
available and falls back to CPU with int8.

**Retries.**
A job is retried up to 3 times on unexpected errors (for example running out of
GPU memory). Errors caused by bad input are not retried, since they will fail the
same way again. When the server restarts, jobs that were queued or in progress
are picked up again.

**Simple storage, same shape as production.**
Each job is a JSON file written atomically (write to a temp file, then rename),
and uploads are kept on local disk. This keeps the project runnable with no
external services.

## What I'd change for production

- **Storage:** uploads in S3 (or similar), job metadata in Postgres, transcripts
  as JSON in object storage or a JSONB column.
- **Queue:** replace the in-process thread pool with SQS / Redis + Celery, so API
  servers and GPU workers scale separately and a crash doesn't lose queued work.
- **Very long files:** split at silences into ~10-minute chunks, transcribe chunks
  in parallel across workers, then shift timestamps and stitch the results.
- **Concurrency:** upload directly to S3 with pre-signed URLs so the API isn't
  moving bytes, rate-limit per client, and autoscale workers on queue depth.
- **API:** authentication (API keys), webhooks when a job finishes instead of
  polling, pagination for listing jobs.
- **Observability:** structured logs, metrics for queue depth, job duration and
  real-time factor, and alerts on failure rate.
