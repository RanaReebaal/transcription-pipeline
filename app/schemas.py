from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    confidence: float


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Transcript(BaseModel):
    language: str
    duration: float
    text: str
    segments: list[Segment]


class Job(BaseModel):
    id: str
    filename: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    attempts: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    result: Transcript | None = None
