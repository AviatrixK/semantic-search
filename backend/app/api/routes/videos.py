import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user, require_admin
from app.core.config import settings
from app.core.db import get_db
from app.models import Job, Video
from app.schemas.video import ChunkOut, JobOut, UploadOut, VideoOut
from app.services import retrieval, storage

router = APIRouter(prefix="/api", tags=["videos"])
ALLOWED = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}
ALLOWED_EXT = {"mp4", "webm", "mov", "mkv"}
ACTIVE_STAGES = ("queued", "extracting", "transcribing", "embedding")


def _size(file: UploadFile) -> int:
    f = file.file
    pos = f.tell()
    f.seek(0, 2)
    size = f.tell()
    f.seek(pos)
    return size


@router.post("/videos", response_model=UploadOut, status_code=202)
def upload_video(file: UploadFile, title: str | None = Form(None, max_length=200),
                 admin: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    name = file.filename or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:  # allow-list also keeps client-controlled text out of the storage key
        raise HTTPException(415, f"Unsupported file extension. Allowed: {', '.join(sorted(ALLOWED_EXT))}")
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported type {file.content_type}")
    size = _size(file)
    if size == 0:
        raise HTTPException(400, "Empty file")
    if size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {settings.MAX_UPLOAD_MB} MB)")

    vid = uuid.uuid4()
    key = f"raw/{vid}.{ext}"
    storage.upload_fileobj(file.file, key, file.content_type)

    video = Video(id=vid, title=(title or "").strip() or name, storage_key=key, uploaded_by=uuid.UUID(admin.id))
    job = Job(video_id=vid)
    db.add(video)
    db.flush()      # insert the video row first so the job's foreign key is valid
    db.add(job)
    db.commit()

    from app.workers.tasks import ingest_video  # local import keeps API startup light
    ingest_video.delay(str(vid), str(job.id))
    return UploadOut(video_id=str(vid), job_id=str(job.id))


@router.post("/videos/{video_id}/reprocess", response_model=UploadOut, status_code=202)
def reprocess_video(video_id: uuid.UUID, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if not v:
        raise HTTPException(404, "Not found")
    if db.scalar(select(Job.id).where(Job.video_id == video_id, Job.stage.in_(ACTIVE_STAGES)).limit(1)):
        raise HTTPException(409, "This video is already being processed")
    job = Job(video_id=video_id)
    v.status = "uploaded"
    db.add(job)
    db.commit()

    from app.workers.tasks import ingest_video
    ingest_video.delay(str(video_id), str(job.id))  # idempotent: old chunks are replaced, never duplicated
    return UploadOut(video_id=str(video_id), job_id=str(job.id))


@router.get("/videos", response_model=list[VideoOut])
def list_videos(_: CurrentUser = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Video).order_by(Video.created_at.desc())).all()
    return [VideoOut(id=str(v.id), title=v.title, status=v.status, duration_sec=v.duration_sec) for v in rows]


@router.get("/videos/{video_id}/stream")
def stream_url(video_id: uuid.UUID, _: CurrentUser = Depends(current_user), db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if not v:
        raise HTTPException(404, "Not found")
    return {"url": storage.presigned_url(v.storage_key)}


@router.get("/videos/{video_id}/transcript", response_model=list[ChunkOut])
def transcript(video_id: uuid.UUID, _: CurrentUser = Depends(current_user), db: Session = Depends(get_db)):
    if not db.get(Video, video_id):
        raise HTTPException(404, "Not found")
    return retrieval.list_chunks(db, video_id)


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: uuid.UUID, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if not v:
        raise HTTPException(404, "Not found")
    storage.delete(v.storage_key)
    storage.delete(f"transcripts/{v.id}.json")
    db.delete(v)  # chunks + jobs cascade
    db.commit()


@router.get("/jobs/{job_id}", response_model=JobOut)
def job_status(job_id: uuid.UUID, _: CurrentUser = Depends(current_user), db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "Not found")
    return JobOut(id=str(j.id), video_id=str(j.video_id), stage=j.stage, progress=j.progress, error=j.error)
