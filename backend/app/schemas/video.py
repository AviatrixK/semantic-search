from datetime import datetime

from pydantic import BaseModel


class UploadOut(BaseModel):
    video_id: str
    job_id: str


class JobOut(BaseModel):
    id: str
    video_id: str
    stage: str
    progress: int
    error: str | None


class VideoOut(BaseModel):
    id: str
    title: str
    status: str
    duration_sec: int | None
    created_at: datetime


class ChunkOut(BaseModel):
    idx: int
    start_sec: float
    end_sec: float
    text: str


class SearchHit(BaseModel):
    video_id: str
    title: str
    start_sec: float
    end_sec: float
    text: str
    score: float
    highlight: str | None = None
    rrf: float | None = None  # hybrid mode: Reciprocal Rank Fusion score (the ranking key)
    found_by: list[str] | None = None  # hybrid mode: which searches returned it ("vector", "keyword")
    rerank_score: float | None = None  # only with RERANK on
