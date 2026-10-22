"""app.scripts.reprocess on the real test database: the plan per video, the dry run, estimates, the confirmation before spending money,
and above all RESUMABILITY (a failure part-way, then the same command again does only what is left). Providers are fakes."""
import hashlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.models import Chunk, ChunkEmbedding, Job, Video
from app.scripts import reprocess
from app.services import jobevents, pricing, vectorstore
from app.services.embedders import EmbeddingUnavailable, GeminiEmbedder


class FakeHosted(GeminiEmbedder):
    def __init__(self, key="k", fail_on_call=None):
        super().__init__(model="fake-emb", dim=4, api_key=key)
        self.calls, self.fail_on_call = [], fail_on_call

    def embed_documents(self, texts):
        self.calls.append(len(texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise EmbeddingUnavailable("provider went away")
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [h[i] + 1.0 for i in range(4)]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        self.tokens_used += sum(len(t) // 4 + 1 for t in texts)
        return out


def add_video(db, title, transcriber=None, n_chunks=3, duration=1800, with_en=False, status="ready"):
    v = Video(id=uuid.uuid4(), title=title, storage_key=f"raw/{title}.mp4", transcriber=transcriber, duration_sec=duration, status=status)
    db.add(v)
    db.flush()
    for i in range(n_chunks):
        db.add(Chunk(id=uuid.uuid4(), video_id=v.id, idx=i, start_sec=i * 30.0, end_sec=i * 30.0 + 30, text=f"{title} chunk {i} text here",
                     text_native=f"{title} chunk {i} text here", text_en=f"english {title} {i}" if with_en else None))
    db.commit()
    return v


@pytest.fixture(autouse=True)
def sarvam_key():
    with patch.object(settings, "SARVAM_API_KEY", "test-key"):  # the missing-key test blanks it again on purpose
        yield


@pytest.fixture
def lib(api, db_session):
    return db_session


def run(args, *, enqueue=None, embedder=None, answer="y"):
    enqueue = enqueue or MagicMock()
    with patch.object(jobevents, "publish"):
        code = reprocess.main(args, enqueue=enqueue, embedder=embedder or FakeHosted(), confirm=lambda prompt: answer)
    return code, enqueue


def vectors(db, emb):
    db.expire_all()
    return db.query(ChunkEmbedding).filter_by(model=emb.key).count()


# ---------------------------------------------------------------- plan and dry run
def test_dry_run_prints_the_plan_and_estimate_and_changes_nothing(lib, capsys):
    a, b = add_video(lib, "alpha"), add_video(lib, "beta", transcriber="sarvam")
    emb = FakeHosted()
    code, enqueue = run(["--dry-run", "--transcriber", "sarvam"], embedder=emb)
    out = capsys.readouterr().out
    assert code == 0 and "Dry run: nothing was changed." in out
    assert f"{a.id}" in out and "whisper -> sarvam" in out  # alpha must be re-transcribed, beta is already sarvam
    # 0.5 h: ONE job at INR 30/h = 15, plus the text APIs: 39,000 characters x 2 versions at INR 20 per 10K = 156
    assert "0.50 h of audio -> about INR 171: one transcription job at INR 30/h plus the text APIs for roman+en at INR 20 per 10K characters" in out
    assert "assuming ~78,000 characters per hour" in out
    enqueue.assert_not_called()
    assert lib.query(Job).count() == 0 and vectors(lib, emb) == 0 and emb.calls == []


def test_the_plan_tells_transcribe_embed_up_to_date_and_busy_apart(lib):
    emb = FakeHosted()
    old = add_video(lib, "old")  # whisper, no vectors
    new = add_video(lib, "new", transcriber="sarvam")  # sarvam, no vectors yet
    done = add_video(lib, "done", transcriber="sarvam")
    vectorstore.store(lib, emb, [(c.id, "native", [1.0, 0, 0, 0]) for c in lib.query(Chunk).filter_by(video_id=done.id)])
    busy = add_video(lib, "busy")
    lib.add(Job(video_id=busy.id, stage="transcribing"))
    lib.commit()
    plans = {p.title: p for p in (reprocess.make_plan(lib, v, transcriber="sarvam", embedder=emb,
                                                      busy={busy.id}) for v in (old, new, done, busy))}
    assert plans["old"].action == "transcribe" and plans["old"].audio_seconds == 1800
    assert plans["new"].action == "embed" and plans["new"].texts == 3 and plans["new"].tokens > 0
    assert plans["done"].action == "up-to-date"
    assert plans["busy"].action == "busy"
    # without --transcriber nothing is ever re-transcribed: only missing vectors are filled
    assert reprocess.make_plan(lib, old, transcriber=None, embedder=emb, busy=set()).action == "embed"


def test_english_variants_are_embedded_too_so_there_are_two_texts_per_chunk(lib):
    v = add_video(lib, "bilingual", transcriber="sarvam", with_en=True)
    p = reprocess.make_plan(lib, v, transcriber="sarvam", embedder=FakeHosted(), busy=set())
    assert p.texts == 6


# ---------------------------------------------------------------- executing and resuming
def test_embed_only_fills_the_vectors_and_a_second_run_has_nothing_left(lib, capsys):
    v = add_video(lib, "talk", transcriber="sarvam", with_en=True)
    emb = FakeHosted()
    assert run(["--embed-provider", "gemini"], embedder=emb)[0] == 0
    assert vectors(lib, emb) == 6 and "embedded 6 text(s)" in capsys.readouterr().out
    calls_after_first = list(emb.calls)
    code, _ = run(["--embed-provider", "gemini"], embedder=emb)
    assert code == 0 and emb.calls == calls_after_first  # not a single new embedding call
    assert "Nothing to do." in capsys.readouterr().out


def test_a_failure_part_way_keeps_the_finished_batches_and_the_rerun_does_only_the_rest(lib, capsys):
    add_video(lib, "big", transcriber="sarvam", n_chunks=5)
    emb = FakeHosted(fail_on_call=2)
    with patch.object(reprocess, "BATCH", 2):
        code, _ = run(["--embed-provider", "gemini"], embedder=emb)
        assert code == 1 and "FAILED: EmbeddingUnavailable" in capsys.readouterr().out
        assert vectors(lib, emb) == 2  # the first batch of 2 was committed before the failure
        emb.fail_on_call = None
        emb.calls.clear()
        code, _ = run(["--embed-provider", "gemini"], embedder=emb)
    assert code == 0 and vectors(lib, emb) == 5
    assert sum(emb.calls) == 3  # only the 3 texts that were still missing were embedded again: 5 - 2


def test_one_failing_video_does_not_stop_the_others(lib, capsys):
    bad = add_video(lib, "bad", transcriber="sarvam", n_chunks=1)
    good = add_video(lib, "good", transcriber="sarvam", n_chunks=1)
    emb = FakeHosted(fail_on_call=1)
    code, _ = run([], embedder=emb)
    assert code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "embedded 1 text(s)" in out
    assert vectors(lib, emb) == 1  # one of the two worked
    run([], embedder=emb)  # the re-run fixes the other
    assert vectors(lib, emb) == 2


def test_queueing_a_transcription_creates_a_job_and_passes_the_overrides(lib):
    v = add_video(lib, "talk")
    code, enqueue = run(["--transcriber", "sarvam", "--embed-provider", "gemini"])
    assert code == 0
    job = lib.query(Job).filter_by(video_id=v.id).one()
    enqueue.assert_called_once_with(str(v.id), str(job.id), "sarvam", "gemini")
    lib.refresh(v)
    assert v.status == "uploaded"  # shown as Processing until the new chunks replace the old ones


def test_rerunning_while_the_job_is_active_neither_duplicates_it_nor_redoes_finished_videos(lib, capsys):
    pending, finished = add_video(lib, "pending"), add_video(lib, "finished", transcriber="sarvam")
    vectorstore.store(lib, FakeHosted(), [(c.id, "native", [1.0, 0, 0, 0]) for c in lib.query(Chunk).filter_by(video_id=finished.id)])
    lib.commit()
    emb = FakeHosted()
    run(["--transcriber", "sarvam"], embedder=emb)
    assert lib.query(Job).count() == 1
    capsys.readouterr()
    code, enqueue = run(["--transcriber", "sarvam"], embedder=emb)  # the worker has not finished: the job is still active
    assert code == 0 and lib.query(Job).count() == 1
    enqueue.assert_not_called()
    assert "busy" in capsys.readouterr().out
    # the worker finishes: the video now records the new engine, so the next run leaves it alone
    lib.query(Job).update({"stage": "done"})
    lib.query(Video).filter_by(id=pending.id).update({"transcriber": "sarvam"})
    lib.commit()
    vectorstore.store(lib, emb, [(c.id, "native", [1.0, 0, 0, 0]) for c in lib.query(Chunk).filter_by(video_id=pending.id)])
    lib.commit()
    code, enqueue = run(["--transcriber", "sarvam"], embedder=emb)
    assert code == 0 and enqueue.call_count == 0 and "Nothing to do." in capsys.readouterr().out


def test_a_failed_job_leaves_the_video_on_its_old_engine_so_the_next_run_retries_it(lib):
    v = add_video(lib, "flaky")
    run(["--transcriber", "sarvam"])
    lib.query(Job).update({"stage": "failed"})  # the worker failed it: transcriber was never changed
    lib.commit()
    _, enqueue = run(["--transcriber", "sarvam"])
    assert enqueue.call_count == 1 and lib.query(Job).count() == 2


# ---------------------------------------------------------------- targeting, confirmation, safety
def test_video_ids_restrict_the_run_and_unknown_or_malformed_ids_are_refused(lib, capsys):
    a, b = add_video(lib, "a"), add_video(lib, "b")
    code, enqueue = run(["--transcriber", "sarvam", "--video-ids", str(a.id)])
    assert code == 0 and enqueue.call_count == 1 and enqueue.call_args[0][0] == str(a.id)
    code, _ = run(["--video-ids", str(uuid.uuid4())])
    assert code == 1 and "Unknown video id" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run(["--video-ids", "not-a-uuid"])
    code, enqueue = run(["--transcriber", "sarvam", "--video-ids", f"{b.id},{uuid.uuid4()}"])  # comma separated works too
    assert code == 1  # one unknown id: nothing is started
    assert reprocess.parse_ids([f"{a.id},{b.id}", str(a.id)]) == [a.id, b.id, a.id]


def test_paid_work_asks_first_and_no_means_nothing_changes(lib, capsys):
    v = add_video(lib, "talk")
    code, enqueue = run(["--transcriber", "sarvam"], answer="n")
    assert code == 1 and "Cancelled" in capsys.readouterr().out
    enqueue.assert_not_called()
    assert lib.query(Job).count() == 0
    code, enqueue = run(["--transcriber", "sarvam"], answer="y")
    assert code == 0 and enqueue.call_count == 1


def test_yes_skips_the_question_and_no_terminal_means_cancel(lib):
    add_video(lib, "talk")
    with patch.object(jobevents, "publish"):
        asked = MagicMock(return_value="n")
        enqueue = MagicMock()
        assert reprocess.main(["--transcriber", "sarvam", "--yes"], enqueue=enqueue, embedder=FakeHosted(), confirm=asked) == 0
        asked.assert_not_called()
        enqueue.reset_mock()

        def eof(prompt):
            raise EOFError

        add_video(lib, "second")
        assert reprocess.main(["--transcriber", "sarvam"], enqueue=enqueue, embedder=FakeHosted(), confirm=eof) == 1  # piped, no answer
        enqueue.assert_not_called()


def test_a_missing_key_stops_before_anything_changes_but_dry_run_still_works(lib, capsys):
    add_video(lib, "talk")
    no_key = FakeHosted(key="")
    code, enqueue = run([], embedder=no_key)
    assert code == 1 and "GEMINI_API_KEY" in capsys.readouterr().out and no_key.calls == []
    assert run(["--dry-run"], embedder=no_key)[0] == 0
    with patch.object(settings, "SARVAM_API_KEY", ""):
        code, _ = run(["--transcriber", "sarvam"], embedder=FakeHosted())
        assert code == 1 and "SARVAM_API_KEY" in capsys.readouterr().out


def test_the_local_provider_fills_missing_local_vectors_without_asking(lib):
    from app.services.embedders import LocalEmbedder
    v = add_video(lib, "talk", n_chunks=2)
    assert lib.query(Chunk).filter(Chunk.embedding.is_(None)).count() == 2
    local = LocalEmbedder()
    asked = MagicMock(return_value="n")
    with patch.object(local.__class__, "embed_documents", lambda self, texts: [[0.5] * settings.EMBED_DIM for _ in texts]), \
         patch.object(jobevents, "publish"):
        code = reprocess.main(["--embed-provider", "local"], enqueue=MagicMock(), embedder=local, confirm=asked)
    assert code == 0 and lib.query(Chunk).filter(Chunk.embedding.is_(None)).count() == 0
    asked.assert_not_called()  # free: no confirmation


def test_estimates_use_the_price_table_and_say_when_a_price_is_unknown(lib):
    emb = FakeHosted()
    plans = [reprocess.Plan(uuid.uuid4(), "x", "embed", texts=10, tokens=2_000_000), reprocess.Plan(uuid.uuid4(), "y", "transcribe", audio_seconds=7200)]
    est = reprocess.estimate(plans, "sarvam", emb)
    assert est["audio_hours"] == 2.0 and est["sarvam_inr"] == pytest.approx(2 * 30 + 2 * 78_000 * 2 / 10_000 * 20) and est["embed_tokens"] == 2_000_000
    assert est["embed_price_known"] is False and est["embed_usd"] is None  # "fake-emb" has no price: never invented
    known = GeminiEmbedder(model="gemini-embedding-2", dim=8, api_key="k")
    assert reprocess.estimate(plans, "whisper", known)["embed_usd"] == pytest.approx(2.0 * pricing.EMBEDDING_PRICES["gemini/gemini-embedding-2"].usd_per_million_tokens)
    assert reprocess.estimate(plans, "whisper", known)["sarvam_inr"] == 0
