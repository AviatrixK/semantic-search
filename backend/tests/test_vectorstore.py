"""chunk_embeddings on the real test database (pgvector): mixed dimensions side by side, one partial HNSW index per model that the
planner really uses, the better of the two text variants per chunk, filters, and resumable re-embedding."""
import math
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from app.core.db import engine
from app.models import Chunk, ChunkEmbedding, Video
from app.services import vectorstore
from app.services.embedders import GeminiEmbedder, LocalEmbedder, OpenAIEmbedder

A = GeminiEmbedder(model="m-a", dim=4, api_key="k")  # 4-dimensional model
B = OpenAIEmbedder(model="m-b", dim=6, api_key="k")  # 6-dimensional model, same column


def unit(*xs):
    n = math.sqrt(sum(x * x for x in xs))
    return [x / n for x in xs]


@pytest.fixture
def lib(api, db_session):
    """Two videos; chunk 0 and 1 of video 1 (chunk 0 has an English variant), chunk 0 of the old video 2."""
    v1 = Video(id=uuid.uuid4(), title="Talk one", storage_key="raw/1.mp4")
    v2 = Video(id=uuid.uuid4(), title="Old talk", storage_key="raw/2.mp4", created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db_session.add_all([v1, v2])
    db_session.flush()
    mk = lambda v, i, native, en=None: Chunk(id=uuid.uuid4(), video_id=v.id, idx=i, start_sec=i * 30.0, end_sec=i * 30.0 + 30, text=native,
                                             text_native=native, text_en=en)
    c0, c1, c2 = mk(v1, 0, "मेरा फोन नंबर है", "my phone number is"), mk(v1, 1, "second chunk"), mk(v2, 0, "old chunk")
    db_session.add_all([c0, c1, c2])
    db_session.commit()
    return type("Lib", (), {"v1": v1, "v2": v2, "c0": c0, "c1": c1, "c2": c2, "db": db_session})


def test_models_of_different_sizes_live_in_one_column(lib):
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(1, 0, 0, 0))])
    vectorstore.store(lib.db, B, [(lib.c0.id, "native", unit(1, 0, 0, 0, 0, 0))])
    lib.db.commit()
    dims = dict(lib.db.execute(text("SELECT model, vector_dims(embedding) FROM chunk_embeddings")).all())
    assert dims == {A.key: 4, B.key: 6}


def test_store_is_an_upsert_so_rerunning_changes_nothing(lib):
    rows = [(lib.c0.id, "native", unit(1, 0, 0, 0)), (lib.c1.id, "native", unit(0, 1, 0, 0))]
    assert vectorstore.store(lib.db, A, rows) == 2
    vectorstore.store(lib.db, A, rows)
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(0, 0, 1, 0))])  # a changed vector replaces the old one
    lib.db.commit()
    assert lib.db.query(ChunkEmbedding).count() == 2
    got = lib.db.query(ChunkEmbedding).filter_by(chunk_id=lib.c0.id).one()
    assert list(got.embedding) == pytest.approx(unit(0, 0, 1, 0))
    assert vectorstore.store(lib.db, A, []) == 0


def test_each_model_gets_its_own_partial_hnsw_index_and_the_planner_uses_it(lib):
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(1, 0, 0, 0)), (lib.c1.id, "native", unit(0, 1, 0, 0))])
    vectorstore.store(lib.db, B, [(lib.c0.id, "native", unit(1, 0, 0, 0, 0, 0))])
    lib.db.commit()
    vectorstore.ensure_index(engine, A)
    vectorstore.ensure_index(engine, A)  # idempotent
    vectorstore.ensure_index(engine, B)
    names = {r[0] for r in lib.db.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk_embeddings'"))}
    assert vectorstore.index_name(A) in names and vectorstore.index_name(B) in names and vectorstore.index_name(A) != vectorstore.index_name(B)
    # On a handful of rows with stale statistics the planner may legitimately pick the plain btree on `model` plus a sort. Take both
    # of those off the table: only an index that returns rows already in distance order is left, which is the HNSW one.
    lib.db.execute(text("SET LOCAL enable_seqscan = off"))
    lib.db.execute(text("SET LOCAL enable_sort = off"))
    plan = "\n".join(r[0] for r in lib.db.execute(text(
        f"EXPLAIN SELECT chunk_id FROM chunk_embeddings WHERE model = '{A.key}' "
        "ORDER BY embedding::vector(4) <=> CAST('[1,0,0,0]' AS vector(4)) LIMIT 5")))
    assert vectorstore.index_name(A) in plan, plan  # the query shape of `nearest` matches the index
    assert vectorstore.index_name(B) not in plan


def test_the_local_model_needs_no_index_here(lib):
    names = lambda: {r[0] for r in lib.db.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk_embeddings'"))}
    before = names()  # (indexes made by other tests stay: TRUNCATE does not drop them)
    assert LocalEmbedder().uses_legacy_column
    vectorstore.ensure_index(engine, LocalEmbedder())  # a no-op: its vectors stay in chunks.embedding
    assert names() == before


def test_nearest_ranks_by_cosine_and_reports_the_matched_variant(lib):
    vectorstore.store(lib.db, A, [
        (lib.c0.id, "native", unit(1, 0, 0, 0)), (lib.c0.id, "en", unit(0, 1, 0, 0)),  # the English variant is the closer one for this query
        (lib.c1.id, "native", unit(1, 1, 0, 0)), (lib.c2.id, "native", unit(0, 0, 1, 0))])
    lib.db.commit()
    hits = vectorstore.nearest(lib.db, A, unit(0, 1, 0, 0), limit=10)
    assert [h["chunk_id"] for h in hits][:2] == [lib.c0.id, lib.c1.id]
    assert hits[0]["matched"] == "en" and hits[0]["score"] == pytest.approx(1.0, abs=1e-4)  # best of its two variants
    assert hits[1]["score"] == pytest.approx(math.sqrt(0.5), abs=1e-3)
    assert len({h["chunk_id"] for h in hits}) == len(hits)  # a chunk appears once even though it has two vectors
    assert hits[0]["title"] == "Talk one" and hits[0]["video_id"] == str(lib.v1.id) and hits[0]["text"] == "मेरा फोन नंबर है"


def test_nearest_only_sees_its_own_models_vectors(lib):
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(1, 0, 0, 0))])
    vectorstore.store(lib.db, B, [(lib.c1.id, "native", unit(1, 0, 0, 0, 0, 0))])
    lib.db.commit()
    assert [h["chunk_id"] for h in vectorstore.nearest(lib.db, A, unit(1, 0, 0, 0), 10)] == [lib.c0.id]
    assert [h["chunk_id"] for h in vectorstore.nearest(lib.db, B, unit(1, 0, 0, 0, 0, 0), 10)] == [lib.c1.id]


def test_nearest_filters_by_video_and_upload_date_and_limits_after_dedupe(lib):
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(1, 0, 0, 0)), (lib.c0.id, "en", unit(1, 0.1, 0, 0)),
                                  (lib.c1.id, "native", unit(1, 0.2, 0, 0)), (lib.c2.id, "native", unit(1, 0.3, 0, 0))])
    lib.db.commit()
    q = unit(1, 0, 0, 0)
    assert {h["video_id"] for h in vectorstore.nearest(lib.db, A, q, 10, video_id=lib.v2.id)} == {str(lib.v2.id)}
    from datetime import date
    assert {h["video_id"] for h in vectorstore.nearest(lib.db, A, q, 10, uploaded_after=date(2021, 1, 1))} == {str(lib.v1.id)}
    assert len(vectorstore.nearest(lib.db, A, q, limit=2)) == 2  # c0's two variants do not use up the limit


def test_a_query_vector_of_the_wrong_size_is_an_error_not_garbage(lib):
    with pytest.raises(ValueError, match="dimensions"):
        vectorstore.nearest(lib.db, A, [1.0, 0.0], 5)


def test_missing_lists_only_the_gaps_which_makes_reembedding_resumable(lib):
    gaps = vectorstore.missing(lib.db, A)
    assert {(cid, f) for cid, f, _ in gaps} == {(lib.c0.id, "native"), (lib.c0.id, "en"), (lib.c1.id, "native"), (lib.c2.id, "native")}
    assert dict(((cid, f), t) for cid, f, t in gaps)[(lib.c0.id, "en")] == "my phone number is"
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(1, 0, 0, 0)), (lib.c0.id, "en", unit(0, 1, 0, 0))])
    lib.db.commit()
    assert {(cid, f) for cid, f, _ in vectorstore.missing(lib.db, A)} == {(lib.c1.id, "native"), (lib.c2.id, "native")}
    assert {(cid, f) for cid, f, _ in vectorstore.missing(lib.db, A, [lib.v2.id])} == {(lib.c2.id, "native")}  # per-video
    assert len(vectorstore.missing(lib.db, B)) == 4  # another model starts from zero
    assert vectorstore.counts_by_model(lib.db) == {A.key: 2}


def test_deleting_a_chunk_removes_its_vectors(lib):
    vectorstore.store(lib.db, A, [(lib.c0.id, "native", unit(1, 0, 0, 0))])
    lib.db.commit()
    lib.db.execute(text("DELETE FROM chunks"))  # the same bulk delete the worker uses on reprocess
    lib.db.commit()
    assert lib.db.query(ChunkEmbedding).count() == 0


def test_a_model_key_that_could_inject_sql_is_refused(lib):
    evil = GeminiEmbedder(model="x'; DROP TABLE chunks; --", dim=4, api_key="k")
    with pytest.raises(ValueError, match="unsafe"):
        vectorstore.nearest(lib.db, evil, unit(1, 0, 0, 0), 5)
    with pytest.raises(ValueError, match="unsafe"):
        vectorstore.ensure_index(engine, evil)
