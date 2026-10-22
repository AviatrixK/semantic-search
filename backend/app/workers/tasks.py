import os
import tempfile
import time
import uuid

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import OperationalError

from celery.utils.log import get_task_logger

from app.core.db import SessionLocal
from app.core.config import settings
from app.core.db import engine
from app.models import Chunk, Job, Video
from app.services import (chunking, embedders, embedding, jobevents, languages, media, sarvam, sentences, storage, tokens,
                          transcription, vectorstore)
from app.workers.celery_app import celery

log = get_task_logger(__name__)

MAX_RETRIES = 3
PROGRESS_INTERVAL_SEC = 3.0
# User-facing text for jobs.error. Never put paths, tool output or exception reprs here: those go to the log.
GENERIC_FAILURE = "Processing failed unexpectedly. Please try again or contact an administrator."
TRANSIENT_FAILURE = "Processing failed because storage or the database was unreachable. Please try again."


def _set(db, job_id, stage, progress, error=None):
    job = db.get(Job, uuid.UUID(job_id))
    job.stage, job.progress, job.error = stage, progress, error
    db.commit()
    jobevents.publish(jobevents.payload(job_id, job.video_id, stage, progress, error))  # after the commit: never announce unsaved state


# Failures with a short user-facing message of their own that retrying cannot fix: bad media, silence, a bad API key, no credits.
# (SarvamUnavailable and EmbeddingUnavailable are subclasses too, but is_transient() sends those to the retry path first.)
KNOWN_PERMANENT = (media.MediaError, transcription.NoSpeechError, sarvam.SarvamError, embedders.EmbeddingError)


def is_transient(exc: BaseException) -> bool:
    """True for infrastructure hiccups worth retrying (S3/DB connectivity). Bad media and silence never are."""
    if isinstance(exc, (sarvam.SarvamUnavailable, embedders.EmbeddingUnavailable)):  # a hosted service that is down or slow: try again
        return True
    if isinstance(exc, KNOWN_PERMANENT):
        return False
    if isinstance(exc, ClientError):  # 404 NoSuchKey etc. are permanent; only 5xx is worth retrying
        return exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) >= 500
    return isinstance(exc, (BotoCoreError, OperationalError, ConnectionError, TimeoutError))


def user_message(exc: BaseException) -> str:
    """Short, path-free text for jobs.error. Known errors carry their own `user_message`."""
    msg = getattr(exc, "user_message", None)
    if msg:
        return msg
    return TRANSIENT_FAILURE if is_transient(exc) else GENERIC_FAILURE


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
    log.warning("job %s failed (video %s): %s", job_id, video_id, message)
    db.rollback()
    _set(db, job_id, "failed", 0, message[:1000])
    video = db.get(Video, uuid.UUID(video_id))
    if video:
        video.status = "failed"
        db.commit()


def _close_sarvam_job(db, video_id) -> None:
    """After a permanent failure the saved Sarvam job must not be resumed by a manual retry (the admin may have changed the language
    or the file): mark it consumed. Jobs Sarvam itself reported as failed stay marked failed."""
    video = db.get(Video, uuid.UUID(video_id))
    if video and video.sarvam_job_id and video.sarvam_job_status != "failed":
        video.sarvam_job_status = sarvam.CONSUMED
        db.commit()


def _store_faststart(video, src: str, tmp: str, job_id: str) -> None:
    """MP4/MOV with the index at the end make the browser fetch the end of the file before it can play. Remux (stream copy,
    no re-encode) and replace the stored object in place. Best effort: the original plays fine, only slower to start, so a
    problem here is logged and never fails the job (scripts/remux_faststart.py can retry it later)."""
    try:
        if not media.needs_faststart(src):
            return
        ext = src.rsplit(".", 1)[-1].lower()
        out = media.remux_faststart(src, os.path.join(tmp, f"faststart.{ext}"))
        with open(out, "rb") as f:
            storage.upload_fileobj(f, video.storage_key, media.FASTSTART_TYPES[ext])
        log.info("job %s: stored a faststart copy of %s", job_id, video.storage_key)
    except Exception as exc:
        log.warning("job %s: faststart remux skipped (%s: %s)", job_id, type(exc).__name__, getattr(exc, "raw", exc))


def _transcribe(db, video, src: str, tmp: str, job_id: str, engine_name: str):
    """Audio out of the video, then speech to text with the chosen engine. Returns (segments, language, billed_audio_seconds)."""
    if engine_name == "sarvam":
        audio = media.extract_audio_mp3(src, os.path.join(tmp, "audio.mp3"))  # raises NoAudioStreamError / MediaError
        video.duration_sec = media.duration_sec(src)
        _store_faststart(video, src, tmp, job_id)
        _set(db, job_id, "transcribing", 25)
        # One Sarvam job per video. Its id is saved on the video (and committed) the moment it exists, and a retry hands it back, so a
        # failure after creation resumes polling the same job instead of paying for a second one.
        def save_job(sarvam_job_id: str, status: str) -> None:
            video.sarvam_job_id, video.sarvam_job_status = sarvam_job_id, status
            db.commit()

        result = sarvam.transcribe_audio(audio, tmp, language=video.language_hint, on_progress=progress_reporter(db, job_id),
                                         job_id=video.sarvam_job_id, job_status=video.sarvam_job_status, on_job=save_job)
        return result.segments, result.language, result.billed_seconds
    wav = media.extract_audio(src, os.path.join(tmp, "audio.wav"))  # raises NoAudioStreamError / MediaError
    video.duration_sec = media.duration_sec(src)
    _store_faststart(video, src, tmp, job_id)
    _set(db, job_id, "transcribing", 25)
    segments, lang = transcription.transcribe(wav, on_progress=progress_reporter(db, job_id),
                                              language=languages.whisper_code(video.language_hint))
    return segments, lang, 0.0


def _ingest(db, video_id, job_id, transcriber: str | None = None, embed_provider: str | None = None):
    video = db.get(Video, uuid.UUID(video_id))
    if video is None:  # deleted while queued
        log.warning("video %s no longer exists, skipping", video_id)
        return
    engine_name = transcriber or settings.TRANSCRIBER
    embedder = embedders.get_embedder(embed_provider)
    embedder.check_configured()  # a missing API key should fail the job NOW, not after minutes of transcription
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, os.path.basename(video.storage_key))
        storage.download_to(video.storage_key, src)

        _set(db, job_id, "extracting", 10)
        segments, lang, billed = _transcribe(db, video, src, tmp, job_id, engine_name)
    video.language = lang
    video.transcriber = engine_name
    storage.put_json(f"transcripts/{video.id}.json", {"video_id": str(video.id), "language": lang, "transcriber": engine_name,
                                                      "segments": segments})

    _set(db, job_id, "embedding", 70)
    spans = chunking.window(segments)
    if not spans:
        raise transcription.NoSpeechError("no chunks produced from transcript")
    text_chars = {}
    if engine_name == "sarvam":  # roman / English versions of each chunk, made from its native text by Sarvam's text APIs
        text_chars = sarvam.add_text_variants(spans, lang)["chars"]
    chunks = [Chunk(id=uuid.uuid4(), video_id=video.id, idx=i, start_sec=s.start, end_sec=s.end, text=s.text, text_native=s.text,
                    text_roman=s.text_roman, text_en=s.text_en) for i, s in enumerate(spans)]
    sentence_rows, vector_rows = [], []
    # All model/provider work happens before touching the DB, so a failure here never leaves half-written chunks.
    if embedder.uses_legacy_column:  # the local model: vectors in chunks.embedding, sentence vectors for highlights
        for c, v in zip(chunks, embedder.embed_documents([c.text for c in chunks])):
            c.embedding = v
        sentence_rows = sentences.build_rows([(c.id, c.text) for c in chunks])
    else:  # a hosted provider: vectors for each text variant in chunk_embeddings (highlights are lexical, no sentence vectors)
        vectorstore.ensure_index(engine, embedder)
        texts = {c.id: vectorstore.field_texts(c) for c in chunks}
        for field in vectorstore.FIELDS:
            ids = [cid for cid, t in texts.items() if field in t]
            for cid, v in zip(ids, embedder.embed_documents([texts[cid][field] for cid in ids]) if ids else []):
                vector_rows.append((cid, field, v))
    # Idempotent: delete + insert + status flip all commit together in the final _set, so a retry or a
    # reprocess can never leave duplicate or half-written chunks (old sentences and vectors go with their chunks: ON DELETE CASCADE).
    db.query(Chunk).filter(Chunk.video_id == video.id).delete()
    db.add_all(chunks)
    db.flush()  # no relationship(), so SQLAlchemy does not order the inserts: chunks must exist first
    db.add_all(sentence_rows)
    vectorstore.store(db, embedder, vector_rows)
    video.status = "ready"
    if video.sarvam_job_id and video.sarvam_job_status != "failed":
        video.sarvam_job_status = sarvam.CONSUMED  # its transcript is in the chunks: a later reprocess must start a new job
    log.info("job %s: done transcriber=%s embedder=%s chunks=%d billed_audio_s=%.0f text_api_chars=%s embed_tokens=%d (estimated %d)",
             job_id, engine_name, embedder.key, len(chunks), billed, text_chars or 0, getattr(embedder, "tokens_used", 0),
             getattr(embedder, "tokens_estimated", 0))
    _set(db, job_id, "done", 100)


@celery.task(bind=True, max_retries=MAX_RETRIES)
def ingest_video(self, video_id: str, job_id: str, transcriber: str | None = None, embed_provider: str | None = None):
    """`transcriber` / `embed_provider` override TRANSCRIBER / EMBED_PROVIDER for this run (app.scripts.reprocess uses them to switch
    engines video by video)."""
    db = SessionLocal()
    try:
        _ingest(db, video_id, job_id, transcriber, embed_provider)
    except Exception as exc:
        if isinstance(exc, KNOWN_PERMANENT) and not is_transient(exc):
            # Permanent (silent audio, corrupt file, no audio track, bad API key): fail the job, never retry.
            # The raw tool output stays in the log; only the short user_message reaches jobs.error.
            log.warning("job %s: %s: %s", job_id, type(exc).__name__, getattr(exc, "raw", exc))
            _fail(db, video_id, job_id, exc.user_message)
            _close_sarvam_job(db, video_id)
            return
        log.exception("job %s: %s", job_id, type(exc).__name__)
        if is_transient(exc) and self.request.retries < self.max_retries:
            db.rollback()
            n = self.request.retries + 1
            log.warning("job %s: transient error, retry %s/%s in %ss", job_id, n, self.max_retries,
                        retry_delay(self.request.retries))
            _set(db, job_id, "queued", 0, f"Temporary problem, retrying ({n}/{self.max_retries}).")
            raise self.retry(exc=exc, countdown=retry_delay(self.request.retries))
        _fail(db, video_id, job_id, user_message(exc))
        raise
    finally:
        db.close()


@celery.task
def purge_expired_tokens() -> int:
    """Periodic (Celery beat, hourly): delete expired refresh tokens."""
    db = SessionLocal()
    try:
        n = tokens.purge_expired_refresh_tokens(db)
        log.info("purged %s expired refresh token(s)", n)
        return n
    finally:
        db.close()
