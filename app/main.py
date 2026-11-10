import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app import formats
from app.audio import ALLOWED_EXTENSIONS, InvalidAudioError, probe
from app.jobs import JobStore, Worker
from app.schemas import Job, JobStatus
from app.transcriber import build_transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per request.
    app.state.store = JobStore(DATA_DIR)
    app.state.worker = Worker(app.state.store, build_transcriber(),
                              max_workers=int(os.getenv("WORKERS", "2")))
    app.state.worker.resume_unfinished()
    yield
    app.state.worker.pool.shutdown(wait=False)


app = FastAPI(title="Transcription Pipeline", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcriptions", status_code=202, response_model=Job)
def create_transcription(file: UploadFile):
    store: JobStore = app.state.store
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    job = store.create(file.filename)
    dest = store.upload_path(job)

    # Stream to disk in chunks and enforce the size limit as we go.
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)

    try:
        probe(dest)
    except InvalidAudioError as exc:
        dest.unlink(missing_ok=True)
        job.status, job.error = JobStatus.failed, str(exc)
        store.save(job)
        raise HTTPException(400, str(exc))

    app.state.worker.submit(job)
    return job


@app.get("/transcriptions/{job_id}", response_model=Job)
def get_transcription(job_id: str):
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/transcriptions/{job_id}/export", response_class=PlainTextResponse)
def export_transcription(job_id: str, format: str = "srt"):
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}")
    exporters = {"srt": formats.to_srt, "vtt": formats.to_vtt, "txt": formats.to_text}
    if format not in exporters:
        raise HTTPException(400, f"format must be one of {list(exporters)}")
    return exporters[format](job.result)
