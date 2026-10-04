"""POST /api/ask: route behaviour with a fake LLM (unit), then the same endpoint against the real test database and
Redis rate limiter (integration). The embedding model and Gemini are never touched."""
import math
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.api.deps import CurrentUser, current_user
from app.core.config import settings
from app.core.db import get_db
from app.core.ratelimit import ask_rate_limit
from app.main import app
from app.models import Chunk, Video
from app.services import llm, rag, retrieval
from app.services.llm import LLMEmptyResponse, LLMNotConfigured, LLMRejected, LLMUnavailable
from tests.fakes import FakeLLM, hit


@pytest.fixture(autouse=True)
def restore_llm():
    yield
    llm.set_llm(None)


# ---------------------------------------------------------------- route (no DB)
@pytest.fixture
def route():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")
    app.dependency_overrides[ask_rate_limit] = lambda: None
    with patch.object(retrieval, "log_query") as log_query, \
         patch.object(retrieval, "vector_search", return_value=[hit(1, title="How to Speak", start=283.0), hit(2)]) as search:
        c = TestClient(app)
        c.log_query, c.search = log_query, search
        yield c
    app.dependency_overrides.clear()


def test_ask_returns_answer_citations_and_mode(route):
    llm.set_llm(FakeLLM("Use rhythm [1]. Pause more [2]. Invented [9]."))
    r = route.post("/api/ask", json={"question": "How do I sound sure?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "rag"
    assert body["answer"] == "Use rhythm [1]. Pause more [2]. Invented."
    assert body["citations"][0] == {"n": 1, "video_id": "11111111-1111-1111-1111-111111111111", "title": "How to Speak",
                                    "start_sec": 283.0, "end_sec": 313.0}
    assert [c["n"] for c in body["citations"]] == [1, 2]
    assert set(body) == {"answer", "citations", "mode"}


def test_ask_logs_the_query_with_mode_ask(route):
    llm.set_llm(FakeLLM("ok [1]"))
    route.post("/api/ask", json={"question": "  How   do I sound   sure? "})
    (_, user_id, question, mode), _ = route.log_query.call_args
    assert mode == "ask" and question == "How do I sound sure?" and uuid.UUID(user_id)


def test_ask_without_matching_context_does_not_call_the_model(route):
    route.search.return_value = []
    fake = FakeLLM("never")
    llm.set_llm(fake)
    r = route.post("/api/ask", json={"question": "Airspeed of a swallow?"})
    assert r.status_code == 200 and r.json() == {"answer": rag.NO_CONTEXT_ANSWER, "citations": [], "mode": "rag"}
    assert fake.calls == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "hi"}, {"question": "   "}, {"question": "x" * 1001},
                                     {"question": 123}, {"q": "How do I sound sure?"}])
def test_ask_validates_the_question(route, payload):
    llm.set_llm(FakeLLM())
    assert route.post("/api/ask", json=payload).status_code == 422


def test_ask_accepts_the_longest_allowed_question(route):
    llm.set_llm(FakeLLM("ok [1]"))
    assert route.post("/api/ask", json={"question": "x" * 1000}).status_code == 200


@pytest.mark.parametrize("error", [LLMUnavailable("api down key=SECRET"), LLMNotConfigured("no key"), LLMRejected("404 model SECRET"),
                                   LLMEmptyResponse("finish_reason=SAFETY")])
def test_llm_failures_become_503_with_a_safe_message(route, error):
    llm.set_llm(FakeLLM(error))
    r = route.post("/api/ask", json={"question": "How do I sound sure?"})
    assert r.status_code == 503
    assert r.json() == {"detail": error.user_message}
    assert "SECRET" not in r.text


def test_ask_requires_login():
    app.dependency_overrides.clear()
    assert TestClient(app).post("/api/ask", json={"question": "How do I sound sure?"}).status_code == 401


# ---------------------------------------------------------------- real DB + Redis
DIM = settings.EMBED_DIM
QUERY = [1.0] + [0.0] * (DIM - 1)


def vec(score):
    return [score, math.sqrt(1 - score * score)] + [0.0] * (DIM - 2)


@pytest.fixture
def seeded(api, db_session):
    v = Video(id=uuid.uuid4(), title="Public speaking", storage_key="raw/a.mp4")
    db_session.add(v)
    db_session.flush()
    db_session.add_all([
        Chunk(id=uuid.uuid4(), video_id=v.id, idx=0, start_sec=0, end_sec=30, text="Slow down and pause between ideas.", embedding=vec(0.9)),
        Chunk(id=uuid.uuid4(), video_id=v.id, idx=1, start_sec=100, end_sec=130, text="Vary your pitch to keep attention.", embedding=vec(0.7)),
        Chunk(id=uuid.uuid4(), video_id=v.id, idx=2, start_sec=200, end_sec=230, text="Unrelated tangent.", embedding=vec(0.1)),
    ])
    db_session.commit()
    api.video_id = str(v.id)
    with patch.object(retrieval, "embed_query", return_value=QUERY):
        yield api


def login(api, email="ann@example.com"):
    api.post("/auth/register", json={"email": email, "password": "Passw0rd123"})
    token = api.post("/auth/login", json={"email": email, "password": "Passw0rd123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_ask_end_to_end_against_pgvector(seeded):
    fake = FakeLLM("Pause between ideas [1] and vary pitch [2]. Made up [8].")
    llm.set_llm(fake)
    r = seeded.post("/api/ask", json={"question": "How should I pace myself?"}, headers=login(seeded))
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Pause between ideas [1] and vary pitch [2]. Made up."
    assert [(c["n"], c["video_id"], c["title"], c["start_sec"]) for c in body["citations"]] == [
        (1, seeded.video_id, "Public speaking", 0.0), (2, seeded.video_id, "Public speaking", 100.0)]
    prompt = fake.calls[0]["prompt"]
    assert "[1] (Public speaking @ 00:00) Slow down and pause between ideas." in prompt
    assert "[2] (Public speaking @ 01:40) Vary your pitch" in prompt
    assert "Unrelated tangent" not in prompt  # below MIN_SCORE: never shown to the model


def test_ask_is_logged_as_mode_ask_and_does_not_count_as_a_search(seeded):
    llm.set_llm(FakeLLM("ok [1]"))
    seeded.post("/api/ask", json={"question": "How should I pace myself?"}, headers=login(seeded))
    from app.core.db import engine
    with engine.connect() as conn:
        modes = conn.execute(text("SELECT mode, query FROM search_logs")).all()
    assert modes == [("ask", "How should I pace myself?")]


def test_ask_is_rate_limited_to_10_per_minute_per_user(seeded):
    llm.set_llm(FakeLLM("ok [1]"))
    alice, bob = login(seeded, "alice@example.com"), login(seeded, "bob@example.com")
    codes = [seeded.post("/api/ask", json={"question": "How should I pace myself?"}, headers=alice).status_code for _ in range(10)]
    assert codes == [200] * 10
    blocked = seeded.post("/api/ask", json={"question": "How should I pace myself?"}, headers=alice)
    assert blocked.status_code == 429 and 1 <= int(blocked.headers["Retry-After"]) <= 60
    assert seeded.post("/api/ask", json={"question": "How should I pace myself?"}, headers=bob).status_code == 200
    assert len(llm.get_llm().calls) == 11  # the blocked request never reached the model


def test_ask_with_nothing_relevant_in_the_database_skips_the_model(seeded):
    fake = FakeLLM("never")
    llm.set_llm(fake)
    with patch.object(retrieval, "embed_query", return_value=[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)):
        r = seeded.post("/api/ask", json={"question": "Something unrelated entirely"}, headers=login(seeded))
    assert r.json()["answer"] == rag.NO_CONTEXT_ANSWER and r.json()["citations"] == []
    assert fake.calls == []
