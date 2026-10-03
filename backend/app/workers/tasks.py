import logging
import os
import tempfile
import time
import uuid

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import OperationalError

from app.core.db import SessionLocal
from app.models import Chunk, Job, Video
from app.services import chunking, embedding, media, storage, transcription
from app.workers.celery_app import celery

log = logging.getLogger(__name__)

MAX_RETRIES = 3
PROGRESS_INTERVAL_SEC = 3.0
NO_SPEECH_MESSAGE = "No speech detected"


def _set(db, job_id, stage, progress, error=None):
    job = db.get(Job, uuid.UUID(job_id))
    job.stage, job.progress, job.error = stage, progress, error
    db.commit()


def is_transient(exc: BaseException) -> bool:
    """True for infrastructure hiccups worth retrying (S3/DB connectivity). Bad media and silence never are."""
    if isinstance(exc, (media.MediaError, transcription.NoSpeechError)):
        return False
    if isinstance(exc, ClientError):  # 404 NoSuchKey etc. are permanent; only 5xx is worth retrying
        return exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) >= 500
    return isinstance(exc, (BotoCoreError, OperationalError, ConnectionError, TimeoutError))


def retry_delay(retries: int) -> int:
    """Exponential backoff: 10s, 20s, 40s ... capped at 5 minutes."""
    return min(10 * 2 ** retries, 300)


def progress_reporter(db, job_id, interval=PROGRESS_INTERVAL_SEC, clock=time.monotonic):
    """Maps transcription position onto job progress 25..70, writing to the DB at most once per `interval`."""
    last = [None]

    def report(done_sec: float, total_sec: float):
        now = clock()
        if last[0] is not None and now - last[0] < interval:
            return
        last[0] = now
        frac = min(max(done_sec / total_sec, 0.0), 1.0) if total_sec > 0 else 0.0
        _set(db, job_id, "transcribing", 25 + int(45 * frac))

    return report


def _fail(db, video_id, job_id, message):
    db.rollback()
    _set(db, job_id, "failed", 0, message[:1000])
    video = db.get(Video, uuid.UUID(video_id))
    if video:
        video.status = "failed"
        db.commit()


def _ingest(db, video_id, job_id):
    video = db.get(Video, uuid.UUID(video_id))
    if video is None:  # deleted while queued
        log.warning("video %s no longer exists, skipping", video_id)
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, os.path.basename(video.storage_key))
        storage.download_to(video.storage_key, src)

        _set(db, job_id, "extracting", 10)
        wav = media.extract_audio(src, os.path.join(tmp, "audio.wav"))  # raises NoAudioStreamError / MediaError
        video.duration_sec = media.duration_sec(src)

        _set(db, job_id, "transcribing", 25)
        segments, lang = transcription.transcribe(wav, on_progress=progress_reporter(db, job_id))
    video.language = lang
    storage.put_json(f"transcripts/{video.id}.json", {"video_id": str(video.id), "language": lang,
                                                      "segments": segments})

    _set(db, job_id, "embedding", 70)
    spans = chunking.window(segments)
    if not spans:
        raise transcription.NoSpeechError(NO_SPEECH_MESSAGE)
    vectors = embedding.embed_batch([s.text for s in spans])
    # Idempotent: delete + insert + status flip all commit together in the final _set, so a retry or a
    # reprocess can never leave duplicate or half-written chunks.
    db.query(Chunk).filter(Chunk.video_id == video.id).delete()
    db.add_all(Chunk(video_id=video.id, idx=i, start_sec=s.start, end_sec=s.end, text=s.text, embedding=v)
               for i, (s, v) in enumerate(zip(spans, vectors)))
    video.status = "ready"
    _set(db, job_id, "done", 100)


@celery.task(bind=True, max_retries=MAX_RETRIES)
def ingest_video(self, video_id: str, job_id: str):
    db = SessionLocal()
    try:
        _ingest(db, video_id, job_id)
    except transcription.NoSpeechError:
        _fail(db, video_id, job_id, NO_SPEECH_MESSAGE)  # permanent: no retry
    except media.MediaError as exc:
        _fail(db, video_id, job_id, str(exc))  # permanent: bad or audio-less media, no retry
    except Exception as exc:
        if is_transient(exc) and self.request.retries < self.max_retries:
            db.rollback()
            n = self.request.retries + 1
            _set(db, job_id, "queued", 0, f"Temporary error, retrying ({n}/{self.max_retries}): {exc}"[:1000])
            raise self.retry(exc=exc, countdown=retry_delay(self.request.retries))
        _fail(db, video_id, job_id, str(exc))
        raise
    finally:
        db.close()
