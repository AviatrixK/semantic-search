"""The ingest pipeline with a hosted transcriber (Sarvam) and a hosted embedder, on the real test database: the native transcript comes
from ONE speech-to-text job, the romanised and English text of each chunk from the text APIs, vectors land in chunk_embeddings (native and
English), no per-sentence vectors are made, the language hint reaches the engine, a retry resumes the saved Sarvam job, and every provider
failure is classified (retry the outage, fail the bad key). Providers are fakes: no network."""
import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from app.core.db import SessionLocal
from app.models import Chunk, ChunkEmbedding, ChunkSentence, Job, Video
from app.services import embedders, jobevents, media, sarvam, storage, transcription, vectorstore
from app.services.embedders import (EmbeddingNotConfigured, EmbeddingRejected, EmbeddingUnavailable, GeminiEmbedder)
from app.services.sarvam import (SarvamJobFailed, SarvamNoTimestamps, SarvamNotConfigured, SarvamRejected, SarvamUnavailable,
                                 TranscriptResult)
from app.workers import tasks
from tests.test_sarvam import FakeSarvam, client_for

REAL_TRANSCRIBE, REAL_ADD_TEXT = sarvam.transcribe_audio, sarvam.add_text_variants  # captured before any fixture patches them

# what the speech-to-text job returns: the native script only
SEGMENTS = [
    {"start": 0.0, "end": 12.0, "text": "मेरा फोन नंबर है"},
    {"start": 12.0, "end": 28.0, "text": "नौ आठ चार शून्य"},
    {"start": 28.0, "end": 55.0, "text": "आज हम सपनों पर बात करेंगे"},
]
# a word-for-word stand-in for the text APIs: word -> (romanised, English)
WORDS = {"मेरा": ("mera", "my"), "फोन": ("phone", "phone"), "नंबर": ("number", "number"), "है": ("hai", "is"), "नौ": ("nau", "nine"),
         "आठ": ("aath", "eight"), "चार": ("chaar", "four"), "शून्य": ("shunya", "zero"), "आज": ("aaj", "today"), "हम": ("hum", "we will"),
         "सपनों": ("sapnon", "dreams"), "पर": ("par", "about"), "बात": ("baat", "talk"), "करेंगे": ("karenge", "")}


def fake_text_variants(spans, language, **kwargs):
    """Stands in for sarvam.add_text_variants: sets text_roman / text_en of every chunk from its native text."""
    fake_text_variants.calls.append((len(spans), language))
    for span in spans:
        span.text_roman = " ".join(WORDS[w][0] for w in span.text.split())
        span.text_en = " ".join(WORDS[w][1] for w in span.text.split() if WORDS[w][1])
    return {"chars": {"transliterate": 1, "translate": 1}, "skipped": []}


fake_text_variants.calls = []


class FakeHosted(GeminiEmbedder):
    def __init__(self, key="k"):
        super().__init__(model="fake-emb", dim=4, api_key=key)
        self.document_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [h[i] + 1.0 for i in range(4)]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out


@pytest.fixture
def world(api, db_session, tmp_path):
    """One uploaded Hindi video with a queued job, and every external service replaced."""
    video = Video(id=uuid.uuid4(), title="Hindi talk", storage_key="raw/h.mp4", language_hint="hi-IN")
    db_session.add(video)
    db_session.flush()
    job = Job(video_id=video.id)
    db_session.add(job)
    db_session.commit()
    emb = FakeHosted()
    sar = MagicMock(return_value=TranscriptResult(segments=[dict(s) for s in SEGMENTS], language="hi-IN", engine="sarvam",
                                                  audio_seconds=55.0, billed_seconds=55.0))
    fake_text_variants.calls.clear()
    (tmp_path / "a.mp3").write_bytes(b"ID3" + b"\x00" * 100)
    puts = []
    with patch.object(storage, "download_to", side_effect=lambda key, path: open(path, "wb").write(b"video")), \
         patch.object(storage, "put_json", side_effect=lambda key, obj: puts.append((key, obj))), \
         patch.object(media, "extract_audio_mp3", return_value=str(tmp_path / "a.mp3")), \
         patch.object(media, "extract_audio", return_value=str(tmp_path / "a.wav")), \
         patch.object(media, "duration_sec", return_value=55), patch.object(media, "needs_faststart", return_value=False), \
         patch.object(sarvam, "transcribe_audio", sar), patch.object(sarvam, "add_text_variants", fake_text_variants), \
         patch.object(embedders, "get_embedder", return_value=emb), \
         patch.object(jobevents, "publish"):
        yield SimpleNamespace(video=video, job=job, emb=emb, sarvam=sar, puts=puts, db=db_session, tmp=tmp_path)


def run(world, transcriber="sarvam"):
    db = SessionLocal()
    try:
        tasks._ingest(db, str(world.video.id), str(world.job.id), transcriber)
    finally:
        db.close()
    world.db.expire_all()


# ---------------------------------------------------------------- the stored result
def test_sarvam_and_a_hosted_embedder_store_three_text_variants_and_two_vectors_per_chunk(world):
    """Native from the one transcription job; romanised and English from the text APIs, per chunk."""
    run(world)
    chunks = world.db.query(Chunk).order_by(Chunk.idx).all()
    assert len(chunks) >= 1
    c0 = chunks[0]
    assert c0.text == c0.text_native and "मेरा फोन नंबर है" in c0.text_native
    assert "mera phone number hai" in c0.text_roman and "my phone number is" in c0.text_en
    assert c0.embedding is None  # hosted vectors never go into the local 384-dim column
    rows = world.db.query(ChunkEmbedding).filter_by(model=world.emb.key).all()
    assert {r.field for r in rows} == {"native", "en"} and len(rows) == 2 * len(chunks)  # roman is searched by keyword, not embedded
    assert all(len(r.embedding) == 4 for r in rows)
    assert world.db.query(ChunkSentence).count() == 0  # no per-sentence vectors for hosted models (highlights are lexical)


def test_the_embedder_gets_the_right_text_for_each_field(world):
    run(world)
    native_texts, en_texts = world.emb.document_calls
    assert all("मेरा" in t or "नौ" in t or "आज" in t for t in native_texts)
    assert all(t.isascii() for t in en_texts) and any("phone" in t for t in en_texts) and any("dreams" in t for t in en_texts)
    assert not any(t.isascii() for t in native_texts)  # the native field is the Devanagari text, not the English one


def test_video_row_records_engine_language_and_status(world):
    run(world)
    v = world.db.get(Video, world.video.id)
    assert (v.transcriber, v.language, v.status) == ("sarvam", "hi-IN", "ready")
    assert world.db.get(Job, world.job.id).stage == "done"
    (key, stored), = world.puts
    assert key == f"transcripts/{world.video.id}.json" and stored["transcriber"] == "sarvam"
    assert stored["segments"][0]["text"] == "मेरा फोन नंबर है" and "text_en" not in stored["segments"][0]  # the stored transcript is native only


def test_the_text_apis_run_once_per_ingest_on_the_chunks_in_the_detected_language(world):
    run(world)
    (n_spans, language), = fake_text_variants.calls
    assert language == "hi-IN" and n_spans == world.db.query(Chunk).count()  # per CHUNK, with the language Sarvam detected


def test_whisper_never_calls_the_text_apis(world):
    segs = [{"start": 0.0, "end": 20.0, "text": "plain english speech here"}]
    with patch.object(transcription, "transcribe", return_value=(segs, "en")):
        run(world, transcriber="whisper")
    assert fake_text_variants.calls == []


def test_the_language_chosen_on_upload_reaches_sarvam(world):
    run(world)
    assert world.sarvam.call_args.kwargs["language"] == "hi-IN"
    world.db.get(Video, world.video.id).language_hint = None
    world.db.commit()
    run(world)
    assert world.sarvam.call_args.kwargs["language"] is None  # no choice = auto-detect


def test_the_transcript_is_searchable_by_all_three_variants_and_by_vector(world):
    run(world)
    from app.services import retrieval
    for query in ("फोन", "mera phone", "dreams"):
        assert retrieval.keyword_search(world.db, query, k=5, highlight=False, embedder=world.emb), query
    chunk_vec = world.emb.embed_documents(["my phone number is nine eight four zero"])[0]
    hits = vectorstore.nearest(world.db, world.emb, chunk_vec, 3)
    assert hits and hits[0]["title"] == "Hindi talk"


def test_reingesting_replaces_everything_with_no_duplicates(world):
    run(world)
    vectors, chunks = world.db.query(ChunkEmbedding).count(), world.db.query(Chunk).count()
    assert vectors > 0 and chunks > 0
    run(world)
    assert (world.db.query(ChunkEmbedding).count(), world.db.query(Chunk).count()) == (vectors, chunks)
    assert world.db.execute(text("SELECT count(*) FROM chunk_embeddings ce LEFT JOIN chunks c ON c.id = ce.chunk_id WHERE c.id IS NULL")).scalar() == 0


def test_the_models_index_exists_after_ingest(world):
    run(world)
    names = {r[0] for r in world.db.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk_embeddings'"))}
    assert vectorstore.index_name(world.emb) in names


# ---------------------------------------------------------------- idempotent transcription: the job id lives on the video
def job_state(world):
    db = SessionLocal()
    try:
        v = db.get(Video, world.video.id)
        return v.sarvam_job_id, v.sarvam_job_status
    finally:
        db.close()


def test_the_job_id_is_committed_the_moment_it_exists_even_if_the_attempt_then_fails(world):
    def create_then_fail(*args, **kwargs):
        kwargs["on_job"]("job-1", "created")
        raise SarvamUnavailable("the upload broke")

    world.sarvam.side_effect = create_then_fail
    with pytest.raises(SarvamUnavailable):
        run(world)
    assert job_state(world) == ("job-1", "created")  # survived the failure (a retry rolls the session back)


def test_a_retry_is_handed_the_saved_job_and_a_finished_ingest_marks_it_consumed(world):
    video = world.db.get(Video, world.video.id)
    video.sarvam_job_id, video.sarvam_job_status = "job-1", "running"
    world.db.commit()
    run(world)
    kwargs = world.sarvam.call_args.kwargs
    assert (kwargs["job_id"], kwargs["job_status"]) == ("job-1", "running") and callable(kwargs["on_job"])
    assert job_state(world) == ("job-1", "consumed")  # kept for the record, never resumed again
    run(world)  # a later reprocess of the same video
    assert world.sarvam.call_args.kwargs["job_status"] == "consumed"  # ... is told it is consumed, so transcribe_audio starts a new job


def test_a_failure_after_job_creation_does_not_create_a_second_job_through_the_whole_pipeline(world):
    """The real sarvam module against a fake Sarvam API. Attempt 1: the job is created, then the upload fails. The worker retries
    the whole ingest. Attempt 2 must continue the SAME job: Sarvam sees exactly one job creation."""
    path = str(world.tmp / "a.mp3")
    out = {"transcript": "x", "language_code": "hi-IN", "timestamps": {
        "chunks": [s["text"] for s in SEGMENTS], "start_time_seconds": [s["start"] for s in SEGMENTS], "end_time_seconds": [s["end"] for s in SEGMENTS]}}
    api = FakeSarvam(output=out)
    api.first_try = True
    api.fail[("POST", "/speech-to-text/job/v1/upload-files")] = lambda body: 503 if api.first_try else 0
    world.sarvam.side_effect = REAL_TRANSCRIBE
    with patch.object(sarvam, "SarvamClient", lambda: client_for(api)), patch.object(media, "duration_float", return_value=55.0),          patch.object(media, "split_audio", return_value=[(path, 0.0)]):
        with pytest.raises(SarvamUnavailable):
            run(world)
        assert api.created() == 1 and job_state(world) == ("job-1", "created")

        api.first_try = False  # the service is back and Celery retries
        run(world)
    assert api.created() == 1 and len(api.jobs) == 1, "a second Sarvam job would be billed again"
    assert api.starts == ["job-1"] and len(api.uploads) == 1
    assert job_state(world) == ("job-1", "consumed")
    assert world.db.get(Video, world.video.id).status == "ready" and world.db.query(Chunk).count() > 0


def test_a_permanent_failure_closes_the_saved_job_so_a_manual_retry_starts_fresh(world):
    video = world.db.get(Video, world.video.id)
    video.sarvam_job_id, video.sarvam_job_status = "job-1", "completed"
    world.db.commit()
    db = SessionLocal()
    try:
        tasks._close_sarvam_job(db, str(world.video.id))
    finally:
        db.close()
    assert job_state(world) == ("job-1", "consumed")
    video = world.db.get(Video, world.video.id)
    video.sarvam_job_status = "failed"
    world.db.commit()
    db = SessionLocal()
    try:
        tasks._close_sarvam_job(db, str(world.video.id))
    finally:
        db.close()
    assert job_state(world) == ("job-1", "failed")  # Sarvam's own verdict is not overwritten


def test_whisper_with_a_hosted_embedder_stores_only_the_native_vector_and_passes_the_language(world):
    segs = [{"start": 0.0, "end": 20.0, "text": "plain english speech here"}, {"start": 20.0, "end": 45.0, "text": "and some more words"}]
    with patch.object(transcription, "transcribe", return_value=(segs, "en")) as whisper:
        run(world, transcriber="whisper")
    assert whisper.call_args.kwargs["language"] == "hi"  # the Sarvam code hi-IN, as Whisper spells it
    assert world.db.query(Video).filter_by(id=world.video.id).one().transcriber == "whisper"
    assert {r.field for r in world.db.query(ChunkEmbedding).all()} == {"native"}
    assert world.db.query(Chunk).first().text_en is None


def test_a_missing_embedding_key_fails_before_any_download_or_transcription(world):
    world.emb._api_key = ""
    with patch.object(storage, "download_to") as download, pytest.raises(EmbeddingNotConfigured):
        run(world)
    download.assert_not_called()
    world.sarvam.assert_not_called()


def test_the_default_provider_is_still_local_so_nothing_changes_until_defaults_are_switched():
    from app.core.config import Settings
    assert (Settings.model_fields["TRANSCRIBER"].default, Settings.model_fields["EMBED_PROVIDER"].default) == ("whisper", "local")


# ---------------------------------------------------------------- failures: retry the outage, fail the bad key
class Retried(Exception):
    pass


@pytest.fixture
def task():
    db = MagicMock()
    with patch.object(tasks, "SessionLocal", return_value=db), patch.object(tasks, "_fail") as fail, \
         patch.object(tasks, "_set") as set_, patch.object(tasks.ingest_video, "retry", side_effect=Retried) as retry:
        yield SimpleNamespace(fail=fail, retry=retry, set=set_)


def run_task(error):
    with patch.object(tasks, "_ingest", side_effect=error):
        return tasks.ingest_video.run(str(uuid.uuid4()), str(uuid.uuid4()))


@pytest.mark.parametrize("error", [SarvamUnavailable("x"), EmbeddingUnavailable("x"), ConnectionError("x")])
def test_an_outage_of_a_hosted_service_is_retried(task, error):
    with pytest.raises(Retried):
        run_task(error)
    task.retry.assert_called_once()
    task.fail.assert_not_called()


@pytest.mark.parametrize("error", [SarvamNotConfigured("x"), SarvamRejected("x"), SarvamJobFailed("x"), SarvamNoTimestamps("x"),
                                   EmbeddingNotConfigured("x"), EmbeddingRejected("x")])
def test_a_bad_key_a_rejected_request_or_a_failed_job_fails_at_once_with_a_readable_message(task, error):
    assert run_task(error) is None  # handled: no exception escapes, so Celery does not retry
    task.retry.assert_not_called()
    message = task.fail.call_args[0][3]
    assert message == error.user_message and 20 < len(message) < 200


def test_the_whole_retry_budget_exhausted_ends_in_a_failed_job_with_the_providers_message(task):
    tasks.ingest_video.push_request(retries=tasks.MAX_RETRIES)
    try:
        with pytest.raises(SarvamUnavailable):
            run_task(SarvamUnavailable("down"))
    finally:
        tasks.ingest_video.pop_request()
    assert task.fail.call_args[0][3] == SarvamUnavailable.user_message
