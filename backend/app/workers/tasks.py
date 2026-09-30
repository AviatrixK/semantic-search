import os
import tempfile
import uuid

from app.core.db import SessionLocal
from app.models import Chunk, Job, Video
from app.services import chunking, embedding, media, storage, transcription
from app.workers.celery_app import celery


def _set(db, job_id, stage, progress, error=None):
    job = db.get(Job, uuid.UUID(job_id))
    job.stage, job.progress, job.error = stage, progress, error
    db.commit()


@celery.task(bind=True, max_retries=2)
def ingest_video(self, video_id: str, job_id: str):
    db = SessionLocal()
    try:
        video = db.get(Video, uuid.UUID(video_id))
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, os.path.basename(video.storage_key))
            storage.download_to(video.storage_key, src)

            _set(db, job_id, "extracting", 10)
            wav = media.extract_audio(src, os.path.join(tmp, "audio.wav"))
            video.duration_sec = media.duration_sec(src)

            _set(db, job_id, "transcribing", 25)
            segments, lang = transcription.transcribe(wav)
            video.language = lang

            _set(db, job_id, "embedding", 70)
            spans = chunking.window(segments)
            vectors = embedding.embed_batch([s.text for s in spans]) if spans else []
            db.query(Chunk).filter(Chunk.video_id == video.id).delete()  # idempotent on retry
            db.add_all(Chunk(video_id=video.id, idx=i, start_sec=s.start, end_sec=s.end, text=s.text, embedding=v)
                       for i, (s, v) in enumerate(zip(spans, vectors)))
            video.status = "ready"
            _set(db, job_id, "done", 100)
    except Exception as exc:
        db.rollback()
        _set(db, job_id, "failed", 0, str(exc)[:1000])
        video = db.get(Video, uuid.UUID(video_id))
        if video:
            video.status = "failed"
            db.commit()
        raise
    finally:
        db.close()
