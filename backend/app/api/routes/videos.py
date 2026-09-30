import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user, require_admin
from app.core.db import get_db
from app.models import Job, Video
from app.schemas.video import JobOut, UploadOut, VideoOut
from app.services import storage

router = APIRouter(prefix="/api", tags=["videos"])
ALLOWED = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}


@router.post("/videos", response_model=UploadOut, status_code=202)
def upload_video(file: UploadFile, admin: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported type {file.content_type}")
    vid = uuid.uuid4()
    ext = (file.filename or "video.mp4").rsplit(".", 1)[-1]
    key = f"raw/{vid}.{ext}"
    storage.upload_fileobj(file.file, key, file.content_type)

    video = Video(id=vid, title=file.filename or str(vid), storage_key=key, uploaded_by=uuid.UUID(admin.id))
    job = Job(video_id=vid)
    db.add_all([video, job])
    db.commit()

    from app.workers.tasks import ingest_video  # local import keeps API startup light
    ingest_video.delay(str(vid), str(job.id))
    return UploadOut(video_id=str(vid), job_id=str(job.id))


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


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: uuid.UUID, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if not v:
        raise HTTPException(404, "Not found")
    storage.delete(v.storage_key)
    db.delete(v)  # chunks + jobs cascade
    db.commit()


@router.get("/jobs/{job_id}", response_model=JobOut)
def job_status(job_id: uuid.UUID, _: CurrentUser = Depends(current_user), db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "Not found")
    return JobOut(id=str(j.id), video_id=str(j.video_id), stage=j.stage, progress=j.progress, error=j.error)
