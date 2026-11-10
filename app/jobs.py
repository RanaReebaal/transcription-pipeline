"""Job storage and background processing.

Uploads return immediately with a job id. A small worker pool does the
transcription so long files don't block the HTTP request. Each job is
saved as JSON on disk, so finished transcripts survive a restart.

For production this would be S3 + Postgres + a real queue (SQS, Celery),
but the interfaces would stay the same.
"""

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.audio import InvalidAudioError, load_audio
from app.schemas import Job, JobStatus, Transcript
from app.transcriber import Transcriber

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def _now():
    return datetime.now(timezone.utc)


class JobStore:
    def __init__(self, data_dir: Path):
        self.uploads = data_dir / "uploads"
        self.jobs_dir = data_dir / "jobs"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename,
                  created_at=_now(), updated_at=_now())
        self.save(job)
        return job

    def upload_path(self, job: Job) -> Path:
        return self.uploads / f"{job.id}{Path(job.filename).suffix.lower()}"

    def save(self, job: Job):
        job.updated_at = _now()
        path = self.jobs_dir / f"{job.id}.json"
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(path)  # atomic, so readers never see half a file

    def get(self, job_id: str) -> Job | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        return Job.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def unfinished(self) -> list[Job]:
        jobs = [self.get(p.stem) for p in self.jobs_dir.glob("*.json")]
        return [j for j in jobs if j and j.status in (JobStatus.queued, JobStatus.processing)]


class Worker:
    def __init__(self, store: JobStore, transcriber: Transcriber, max_workers: int = 2):
        self.store = store
        self.transcriber = transcriber
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, job: Job):
        self.pool.submit(self._run, job.id)

    def resume_unfinished(self):
        # Jobs that were mid-flight when the server stopped get picked up again.
        for job in self.store.unfinished():
            log.info("Resuming job %s", job.id)
            self.submit(job)

    def _run(self, job_id: str):
        job = self.store.get(job_id)
        if job is None:
            return

        while job.attempts < MAX_ATTEMPTS:
            job.attempts += 1
            job.status = JobStatus.processing
            job.progress = 0.0
            self.store.save(job)
            try:
                job.result = self._transcribe(job)
                job.status = JobStatus.completed
                job.progress = 1.0
                job.error = None
                self.store.save(job)
                return
            except InvalidAudioError as exc:
                # Bad input won't get better on retry.
                job.error = str(exc)
                break
            except Exception as exc:
                log.exception("Job %s attempt %d failed", job.id, job.attempts)
                job.error = f"{type(exc).__name__}: {exc}"

        job.status = JobStatus.failed
        self.store.save(job)

    def _transcribe(self, job: Job) -> Transcript:
        audio = load_audio(self.store.upload_path(job))

        def on_progress(p: float):
            job.progress = round(p, 3)
            self.store.save(job)

        language, segments = self.transcriber.transcribe(audio, on_progress)
        segments = list(segments)
        return Transcript(
            language=language,
            duration=round(len(audio) / 16_000, 2),
            text=" ".join(s.text for s in segments),
            segments=segments,
        )
