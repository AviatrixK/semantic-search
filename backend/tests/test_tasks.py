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
    job_id = uuid.uuid4()
    db = MagicMock()
    db.get.side_effect = lambda model, _id: {Video: video, Job: job}[model]
    with patch.object(tasks, "SessionLocal", return_value=db), \
         patch.object(storage, "download_to") as download, \
         patch.object(storage, "put_json") as put_json, \
         patch.object(media, "extract_audio", return_value="audio.wav"), \
         patch.object(media, "duration_sec", return_value=20), \
         patch.object(embedding, "embed_batch", side_effect=lambda t: [[0.0] * 384 for _ in t]) as embed, \
         patch.object(tasks.ingest_video, "retry", side_effect=Retried) as retry:
        yield SimpleNamespace(db=db, video=video, job=job, job_id=job_id, download=download, put_json=put_json, embed=embed,
                              retry=retry)


def run(env):
    return tasks.ingest_video.run(str(env.video.id), str(env.job_id))


def test_success_stores_transcript_and_chunks_idempotently(env):
    with patch.object(transcription, "transcribe", return_value=(SEGMENTS, "en")):
        run(env)
    assert (env.job.stage, env.job.progress, env.job.error) == ("done", 100, None)
    assert env.video.status == "ready" and env.video.language == "en" and env.video.duration_sec == 20
    key, payload = env.put_json.call_args[0]
    assert key == f"transcripts/{env.video.id}.json"
    assert payload["segments"] == SEGMENTS and payload["language"] == "en"
    env.db.query.return_value.filter.return_value.delete.assert_called_once()  # old chunks replaced first
    chunks = list(env.db.add_all.call_args_list[0][0][0])
    assert len(chunks) == 1 and chunks[0].idx == 0 and chunks[0].text == "hello world"
    env.retry.assert_not_called()


def test_ingest_stores_sentence_embeddings_after_their_chunks_exist(env):
    segs = [{"start": 0.0, "end": 5.0, "text": "Hello there."}, {"start": 5.0, "end": 9.0, "text": "How are you?"}]
    with patch.object(transcription, "transcribe", return_value=(segs, "en")):
        run(env)
    names = [c[0] for c in env.db.method_calls if c[0] in ("query", "add_all", "flush")]
    assert names == ["query", "add_all", "flush", "add_all"]  # delete old, insert chunks, flush, insert sentences
    chunks = list(env.db.add_all.call_args_list[0][0][0])
    sentence_rows = list(env.db.add_all.call_args_list[1][0][0])
    assert [(s.idx, s.text) for s in sentence_rows] == [(0, "Hello there."), (1, "How are you?")]
    assert {s.chunk_id for s in sentence_rows} == {chunks[0].id}  # FK target is the chunk inserted just before
    assert env.job.stage == "done"


def test_no_speech_fails_job_with_readable_message_and_no_retry(env):
    with patch.object(transcription, "transcribe", side_effect=transcription.NoSpeechError("x")):
        run(env)  # handled failure: no exception escapes, so Celery does not retry
    assert env.job.stage == "failed" and env.job.error == "No speech detected in this video."
    assert env.video.status == "failed"
    env.retry.assert_not_called()
    env.put_json.assert_not_called()
    env.embed.assert_not_called()
    env.db.add_all.assert_not_called()  # never "done" with zero chunks


RAW_FFPROBE = ("[mov,mp4,m4a,3gp,3g2,mj2 @ 0x7f3a2c001480] moov atom not found\n"
               "/tmp/tmpab12cd/1c2d.mp4: Invalid data found when processing input")


def assert_clean(error):
    assert "/tmp" not in error and "\\" not in error and "0x" not in error and "ffprobe" not in error
    assert "moov" not in error and ".mp4" not in error


def test_corrupt_file_maps_to_short_message_and_is_not_retried(env, caplog):
    err = media.MediaError(f"ffprobe failed (exit 1): {RAW_FFPROBE}", raw=RAW_FFPROBE)
    with patch.object(media, "extract_audio", side_effect=err), caplog.at_level("WARNING"):
        run(env)  # handled failure: nothing escapes, so Celery cannot retry it
    assert env.job.stage == "failed" and env.job.error == "This file is not a valid video."
    assert_clean(env.job.error)
    assert env.video.status == "failed"
    env.retry.assert_not_called()
    assert "moov atom not found" in caplog.text and str(env.job_id) in caplog.text  # raw output logged with job_id


def test_no_audio_track_maps_to_short_message_and_is_not_retried(env):
    with patch.object(media, "extract_audio",
                      side_effect=media.NoAudioStreamError("/tmp/tmpab/x.mp4 has no audio stream")):
        run(env)
    assert env.job.stage == "failed" and env.job.error == "This video has no audio track."
    assert_clean(env.job.error)
    env.retry.assert_not_called()


def test_missing_ffmpeg_does_not_leak_install_hint_or_retry(env):
    with patch.object(media, "extract_audio", side_effect=media.ToolNotFoundError("'ffmpeg' was not found on PATH")):
        run(env)
    assert env.job.stage == "failed" and "PATH" not in env.job.error
    env.retry.assert_not_called()


def test_failed_job_logs_warning_with_ids_and_user_message(env, caplog):
    with patch.object(transcription, "transcribe", side_effect=transcription.NoSpeechError("x")), \
         caplog.at_level("WARNING"):
        run(env)
    rec = [r for r in caplog.records if r.levelname == "WARNING" and "failed (video" in r.getMessage()]
    assert len(rec) == 1
    msg = rec[0].getMessage()
    assert str(env.job_id) in msg and str(env.video.id) in msg and "No speech detected in this video." in msg


def test_user_message_mapping():
    assert tasks.user_message(media.MediaError("raw /tmp/x")) == "This file is not a valid video."
    assert tasks.user_message(media.NoAudioStreamError("raw")) == "This video has no audio track."
    assert tasks.user_message(transcription.NoSpeechError("raw")) == "No speech detected in this video."
    assert tasks.user_message(RuntimeError("boom at 0x7f00 in /app/x.py")) == tasks.GENERIC_FAILURE
    assert tasks.user_message(EndpointConnectionError(endpoint_url="http://minio:9000")) == tasks.TRANSIENT_FAILURE
    for m in (tasks.GENERIC_FAILURE, tasks.TRANSIENT_FAILURE):
        assert_clean(m)


def test_transient_error_retries_with_backoff(env):
    env.download.side_effect = EndpointConnectionError(endpoint_url="http://minio:9000")
    with pytest.raises(Retried):
        run(env)
    assert env.job.stage == "queued" and env.job.error == "Temporary problem, retrying (1/3)."
    assert "minio" not in env.job.error
    assert env.retry.call_args.kwargs["countdown"] == 10
    assert env.video.status == "uploaded"  # not marked failed while retries remain


def test_transient_error_fails_when_retries_exhausted(env):
    env.download.side_effect = EndpointConnectionError(endpoint_url="http://minio:9000")
    with patch.object(tasks.ingest_video, "max_retries", 0), pytest.raises(EndpointConnectionError):
        run(env)
    assert env.job.stage == "failed" and env.video.status == "failed"
    assert env.job.error == tasks.TRANSIENT_FAILURE
    env.retry.assert_not_called()


def test_unexpected_error_fails_without_retry(env):
    with patch.object(transcription, "transcribe", side_effect=RuntimeError("boom")), pytest.raises(RuntimeError):
        run(env)
    assert env.job.stage == "failed" and env.job.error == tasks.GENERIC_FAILURE  # raw text only goes to the log
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
