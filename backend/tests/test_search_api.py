"""/api/search against the real test database and pgvector. The embedding model is never loaded: query and sentence
embeddings are replaced with hand-built vectors whose cosine similarity to the query is known exactly."""
import math
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from app.api.deps import CurrentUser, current_user
from app.core.config import settings
from app.core.db import get_db
from app.core.ratelimit import search_rate_limit
from app.main import app
from app.models import Chunk, ChunkSentence, Video
from app.services import embedding, retrieval, sentences

DIM = settings.EMBED_DIM
QUERY = [1.0] + [0.0] * (DIM - 1)


def vec(score):
    """Unit vector whose cosine similarity with QUERY is exactly `score`."""
    return [score, math.sqrt(1 - score * score)] + [0.0] * (DIM - 2)


def fake_embed_batch(sentences):
    return [vec(0.95 if "robots" in s.lower() else 0.1) for s in sentences]


# ---------------------------------------------------------------- route wiring (no DB)
@pytest.fixture
def route_client():
    from fastapi.testclient import TestClient
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")
    app.dependency_overrides[search_rate_limit] = lambda: None
    with patch.object(retrieval, "log_query"), patch.object(retrieval, "vector_search", return_value=[]) as vs:
        c = TestClient(app)
        c.vs = vs
        yield c
    app.dependency_overrides.clear()


def test_route_passes_filters_and_sets_timing_header(route_client):
    vid = uuid.uuid4()
    r = route_client.get(f"/api/search?q=public+speaking&k=5&video_id={vid}&uploaded_after=2024-01-02&highlight=false")
    assert r.status_code == 200
    ms = float(r.headers["X-Search-Ms"])
    assert 0 <= ms < 5000
    kwargs = route_client.vs.call_args.kwargs
    assert kwargs["k"] == 5 and kwargs["video_id"] == vid and kwargs["highlight"] is False
    assert kwargs["uploaded_after"].isoformat() == "2024-01-02"


def test_route_defaults_and_validation(route_client):
    assert route_client.get("/api/search?q=hello").status_code == 200
    kwargs = route_client.vs.call_args.kwargs
    assert kwargs["video_id"] is None and kwargs["uploaded_after"] is None and kwargs["highlight"] is True
    assert route_client.get("/api/search?q=hello&uploaded_after=not-a-date").status_code == 422
    assert route_client.get("/api/search?q=hello&video_id=nope").status_code == 422
    assert route_client.get("/api/search?q=hello&k=0").status_code == 422
    assert route_client.get("/api/search?q=h").status_code == 422


def test_cors_exposes_timing_header():
    from fastapi.testclient import TestClient
    r = TestClient(app).options("/api/search", headers={"Origin": settings.CORS_ORIGINS[0],
                                                        "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200  # preflight ok
    r = TestClient(app).get("/health", headers={"Origin": settings.CORS_ORIGINS[0]})
    assert "X-Search-Ms" in r.headers["access-control-expose-headers"]


# ---------------------------------------------------------------- end to end on pgvector
@pytest.fixture
def seeded(api, db_session):
    """Video 1 (recent, 78-ish style overlapping chunks) and video 2 (uploaded in 2020)."""
    v1, v2 = Video(id=uuid.uuid4(), title="Public speaking", storage_key="raw/a.mp4"), \
        Video(id=uuid.uuid4(), title="Old talk", storage_key="raw/b.mp4",
              created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db_session.add_all([v1, v2])
    db_session.flush()
    rows = [  # (video, idx, start, end, score, text)
        (v1, 0, 0, 30, 0.90, "Hello everyone. Robots can teach us how to live. Thank you all."),
        (v1, 1, 25, 55, 0.80, "Thank you all. Next point about slides."),  # overlaps idx 0: must be deduped
        (v1, 2, 50, 80, 0.70, "Next point about slides. Keep them simple."),  # overlaps only idx 1: survives
        (v1, 3, 200, 230, 0.20, "Completely unrelated rambling."),  # below MIN_SCORE
        (v1, 4, 300, 330, 0.50, "A single sentence chunk with no punctuation break"),
        (v2, 0, 0, 30, 0.60, "Old video about robots. Nothing else."),
    ]
    chunks = [Chunk(id=uuid.uuid4(), video_id=v.id, idx=i, start_sec=s, end_sec=e, text=t, embedding=vec(score))
              for v, i, s, e, score, t in rows]
    db_session.add_all(chunks)
    db_session.flush()
    with patch.object(embedding, "embed_batch", side_effect=fake_embed_batch):  # what the worker does at ingest
        db_session.add_all(sentences.build_rows([(c.id, c.text) for c in chunks]))
    db_session.commit()
    api.post("/auth/register", json={"email": "ann@example.com", "password": "Passw0rd123"})
    token = api.post("/auth/login", json={"email": "ann@example.com", "password": "Passw0rd123"}).json()["access_token"]
    with patch.object(retrieval, "embed_query", return_value=QUERY), \
         patch.object(retrieval, "embed_batch", side_effect=fake_embed_batch) as eb:
        api.headers.update({"Authorization": f"Bearer {token}"})
        api.v1, api.v2, api.embed_batch = v1, v2, eb
        yield api


def test_threshold_and_dedupe_end_to_end(seeded):
    r = seeded.get("/api/search?q=robots+teach")
    assert r.status_code == 200
    hits = r.json()
    assert [(h["video_id"] == str(seeded.v1.id), h["start_sec"]) for h in hits] == \
        [(True, 0), (True, 50), (False, 0), (True, 300)]  # idx1 deduped, idx3 below threshold
    assert [h["score"] for h in hits] == pytest.approx([0.9, 0.7, 0.6, 0.5], abs=1e-3)
    assert all(h["score"] >= settings.MIN_SCORE for h in hits)
    assert float(r.headers["X-Search-Ms"]) >= 0


def test_k_limits_after_dedupe(seeded):
    assert len(seeded.get("/api/search?q=robots&k=2").json()) == 2


def test_video_filter(seeded):
    hits = seeded.get(f"/api/search?q=robots&video_id={seeded.v2.id}").json()
    assert [h["title"] for h in hits] == ["Old talk"]


def test_uploaded_after_filter(seeded):
    titles = {h["title"] for h in seeded.get("/api/search?q=robots&uploaded_after=2021-01-01").json()}
    assert titles == {"Public speaking"}
    both = {h["title"] for h in seeded.get("/api/search?q=robots&uploaded_after=2020-01-01").json()}
    assert both == {"Public speaking", "Old talk"}  # the boundary day is included
    assert seeded.get("/api/search?q=robots&uploaded_after=2999-01-01").json() == []


def test_highlight_is_the_best_matching_sentence(seeded):
    hits = {h["start_sec"]: h for h in seeded.get(f"/api/search?q=robots&video_id={seeded.v1.id}").json()}
    assert hits[0]["highlight"] == "Robots can teach us how to live."
    assert hits[300]["highlight"] == hits[300]["text"]  # one sentence: nothing to choose from


def test_highlight_uses_stored_sentence_vectors_without_calling_the_model(seeded):
    hits = seeded.get("/api/search?q=robots&k=10").json()
    assert all(h["highlight"] for h in hits)
    assert seeded.embed_batch.call_count == 0  # query-time embedding is what made this 500 ms


def test_highlight_falls_back_to_embedding_when_sentences_are_not_stored(seeded, db_session):
    db_session.execute(text("DELETE FROM chunk_sentences"))  # e.g. ingested before the table existed
    db_session.commit()
    hits = seeded.get(f"/api/search?q=robots&video_id={seeded.v1.id}").json()
    assert {h["start_sec"]: h["highlight"] for h in hits}[0] == "Robots can teach us how to live."
    assert seeded.embed_batch.call_count == 1  # one batch for every sentence that needs it


def test_stale_stored_sentences_are_ignored(seeded, db_session):
    db_session.execute(text("UPDATE chunk_sentences SET text = 'old split rule' WHERE idx = 0"))
    db_session.commit()
    hits = seeded.get(f"/api/search?q=robots&video_id={seeded.v1.id}").json()
    assert {h["start_sec"]: h["highlight"] for h in hits}[0] == "Robots can teach us how to live."
    assert seeded.embed_batch.call_count == 1


def test_search_still_works_if_chunk_sentences_table_is_missing(seeded, db_session):
    """Before the ALTER is applied (e.g. the API reloaded new code first), highlights degrade, search does not."""
    db_session.execute(text("ALTER TABLE chunk_sentences RENAME TO chunk_sentences_off"))
    db_session.commit()
    try:
        r = seeded.get(f"/api/search?q=robots&video_id={seeded.v1.id}")
        assert r.status_code == 200
        assert {h["start_sec"]: h["highlight"] for h in r.json()}[0] == "Robots can teach us how to live."
    finally:  # the schema is shared by the whole test session
        db_session.rollback()
        db_session.execute(text("ALTER TABLE chunk_sentences_off RENAME TO chunk_sentences"))
        db_session.commit()


def test_deleting_chunks_removes_their_sentences(seeded, db_session):
    assert db_session.scalar(text("SELECT count(*) FROM chunk_sentences")) > 0
    db_session.execute(text("DELETE FROM chunks"))  # same bulk delete the worker uses on reprocess
    db_session.commit()
    assert db_session.scalar(text("SELECT count(*) FROM chunk_sentences")) == 0


def test_highlight_can_be_switched_off(seeded):
    hits = seeded.get("/api/search?q=robots&highlight=false").json()
    assert hits and all(h["highlight"] is None for h in hits)
    assert seeded.embed_batch.call_count == 0


def test_nonsense_query_returns_nothing(seeded):
    with patch.object(retrieval, "embed_query", return_value=[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)):
        r = seeded.get("/api/search?q=purple+elephant")
    assert r.status_code == 200 and r.json() == []


def test_search_is_logged_and_requires_login(seeded):
    seeded.get("/api/search?q=robots+teach")
    assert seeded_db_count(seeded) == 1
    seeded.headers.pop("Authorization")
    assert seeded.get("/api/search?q=robots").status_code == 401


def seeded_db_count(api):
    from app.core.db import engine
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM search_logs WHERE mode='search'")).scalar()
