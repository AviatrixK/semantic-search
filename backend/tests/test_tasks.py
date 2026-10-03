import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from sqlalchemy.exc import OperationalError

from app.models import Job, Video
from app.services import embedding, media, storage, transcription
from app.workers import tasks

SEGMENTS = [{"start": 0.0, "end": 10.0, "text": "hello"}, {"start": 10.0, "end": 20.0, "text": "world"}]


class Retried(Exception):
    pass


@pytest.fixture
def env():
    """Fake DB session + patched I/O. Only attributes are patched (never whole modules) so the real exception
    classes still work in the worker's except clauses."""
    video = SimpleNamespace(id=uuid.uuid4(), storage_key="raw/x.mp4", status="uploaded", duration_sec=None,
                            language=None)
    job = SimpleNamespace(stage="queued", progress=0, error=None)
    db = MagicMock()
    db.get.side_effect = lambda model, _id: {Video: video, Job: job}[model]
    with patch.object(tasks, "SessionLocal", return_value=db), \
         patch.object(storage, "download_to") as download, \
         patch.object(storage, "put_json") as put_json, \
         patch.object(media, "extract_audio", return_value="audio.wav"), \
         patch.object(media, "duration_sec", return_value=20), \
         patch.object(embedding, "embed_batch", side_effect=lambda t: [[0.0] * 384 for _ in t]) as embed, \
         patch.object(tasks.ingest_video, "retry", side_effect=Retried) as retry:
        yield SimpleNamespace(db=db, video=video, job=job, download=download, put_json=put_json, embed=embed,
                              retry=retry)


def run(env):
    return tasks.ingest_video.run(str(env.video.id), str(uuid.uuid4()))


def test_success_stores_transcript_and_chunks_idempotently(env):
    with patch.object(transcription, "transcribe", return_value=(SEGMENTS, "en")):
        run(env)
    assert (env.job.stage, env.job.progress, env.job.error) == ("done", 100, None)
    assert env.video.status == "ready" and env.video.language == "en" and env.video.duration_sec == 20
    key, payload = env.put_json.call_args[0]
    assert key == f"transcripts/{env.video.id}.json"
    assert payload["segments"] == SEGMENTS and payload["language"] == "en"
    env.db.query.return_value.filter.return_value.delete.assert_called_once()  # old chunks replaced first
    chunks = list(env.db.add_all.call_args[0][0])
    assert len(chunks) == 1 and chunks[0].idx == 0 and chunks[0].text == "hello world"
    env.retry.assert_not_called()


def test_no_speech_fails_job_with_readable_message_and_no_retry(env):
    with patch.object(transcription, "transcribe", side_effect=transcription.NoSpeechError("x")):
        run(env)  # handled failure: no exception escapes, so Celery does not retry
    assert env.job.stage == "failed" and env.job.error == "No speech detected"
    assert env.video.status == "failed"
    env.retry.assert_not_called()
    env.put_json.assert_not_called()
    env.embed.assert_not_called()
    env.db.add_all.assert_not_called()  # never "done" with zero chunks


def test_bad_media_fails_without_retry(env):
    with patch.object(media, "extract_audio", side_effect=media.NoAudioStreamError("clip.mp4 has no audio stream")):
        run(env)
    assert env.job.stage == "failed" and "no audio stream" in env.job.error
    env.retry.assert_not_called()


def test_transient_error_retries_with_backoff(env):
    env.download.side_effect = EndpointConnectionError(endpoint_url="http://minio:9000")
    with pytest.raises(Retried):
        run(env)
    assert env.job.stage == "queued" and "retrying (1/3)" in env.job.error
    assert env.retry.call_args.kwargs["countdown"] == 10
    assert env.video.status == "uploaded"  # not marked failed while retries remain


def test_transient_error_fails_when_retries_exhausted(env):
    env.download.side_effect = EndpointConnectionError(endpoint_url="http://minio:9000")
    with patch.object(tasks.ingest_video, "max_retries", 0), pytest.raises(EndpointConnectionError):
        run(env)
    assert env.job.stage == "failed" and env.video.status == "failed"
    env.retry.assert_not_called()


def test_unexpected_error_fails_without_retry(env):
    with patch.object(transcription, "transcribe", side_effect=RuntimeError("boom")), pytest.raises(RuntimeError):
        run(env)
    assert env.job.stage == "failed" and env.job.error == "boom"
    env.retry.assert_not_called()


def test_missing_video_is_skipped(env):
    env.db.get.side_effect = lambda model, _id: None
    run(env)
    env.download.assert_not_called()


def client_error(status):
    return ClientError({"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": "X"}}, "GetObject")


@pytest.mark.parametrize("exc,expected", [
    (EndpointConnectionError(endpoint_url="u"), True),
    (OperationalError("select 1", {}, Exception("connection refused")), True),
    (ConnectionError(), True),
    (TimeoutError(), True),
    (client_error(503), True),
    (client_error(404), False),
    (media.MediaError("bad"), False),
    (media.NoAudioStreamError("none"), False),
    (media.ToolNotFoundError("ffmpeg"), False),
    (transcription.NoSpeechError("silent"), False),
    (ValueError("x"), False),
])
def test_is_transient(exc, expected):
    assert tasks.is_transient(exc) is expected


def test_retry_delay_is_exponential_and_capped():
    assert [tasks.retry_delay(n) for n in range(3)] == [10, 20, 40]
    assert tasks.retry_delay(10) == 300


def test_progress_maps_to_25_70_and_is_throttled():
    now = [0.0]
    db = MagicMock()
    job = SimpleNamespace(stage="", progress=0, error=None)
    db.get.return_value = job
    report = tasks.progress_reporter(db, str(uuid.uuid4()), interval=3.0, clock=lambda: now[0])

    report(0, 100)
    assert (job.stage, job.progress) == ("transcribing", 25)
    now[0] = 1.0
    report(50, 100)  # within 3s of the last write: skipped
    assert job.progress == 25 and db.commit.call_count == 1
    now[0] = 3.5
    report(50, 100)
    assert job.progress == 47 and db.commit.call_count == 2
    now[0] = 7.0
    report(100, 100)
    assert job.progress == 70
    now[0] = 11.0
    report(150, 100)  # overshoot is clamped
    assert job.progress == 70
    now[0] = 15.0
    report(5, 0)  # unknown duration must not divide by zero
    assert job.progress == 25
