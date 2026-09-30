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


class SearchHit(BaseModel):
    video_id: str
    title: str
    start_sec: float
    end_sec: float
    text: str
    score: float
