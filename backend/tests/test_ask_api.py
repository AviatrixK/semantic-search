"""POST /api/ask and GET /api/ask/stream: route behaviour with fakes (unit), then the same endpoints against the real test
database and Redis (integration). The embedding model and Gemini are never touched."""
import json
import math
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.agent import orchestrator
from app.api.deps import CurrentUser, current_user
from app.api.routes import ask as ask_routes
from app.core.config import settings
from app.core.db import get_db
from app.core.ratelimit import ask_rate_limit
from app.main import app
from app.models import Chunk, Video
from app.services import llm, rag, retrieval
from app.services.llm import LLMEmptyResponse, LLMNotConfigured, LLMRejected, LLMUnavailable
from app.services.memory import ChatMemory
from app.services.usage import DAY_SECONDS, UsageMeter
from tests.fakes import FakeLLM, FakeRedis, ScriptedLLM, call, hit, text_turn, tools_turn

VID = "11111111-1111-1111-1111-111111111111"  # the video of tests.fakes.hit()
USER = str(uuid.uuid4())
SID = "chat-session-1"
Q = "How do I sound sure?"


@pytest.fixture(autouse=True)
def restore_llm():
    yield
    llm.set_llm(None)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]  # ": keep-alive" comments are ignored
        if lines:
            name = next(line[7:] for line in lines if line.startswith("event: "))
            data = next(line[6:] for line in lines if line.startswith("data: "))
            events.append((name, json.loads(data)))
    return events


# ---------------------------------------------------------------- unit: fakes for DB, search, Redis
@pytest.fixture
def route():
    redis = FakeRedis()
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=USER, role="user")
    app.dependency_overrides[ask_rate_limit] = lambda: None
    app.dependency_overrides[ask_routes.get_session_factory] = lambda: MagicMock  # calling it gives a context-manager-capable mock
    app.dependency_overrides[ask_routes.get_meter] = lambda: UsageMeter(client=redis, budget=50_000)
    app.dependency_overrides[ask_routes.get_memory] = lambda: ChatMemory(client=redis)
    search = MagicMock(return_value=[hit(1, title="How to Speak", start=283.0), hit(2)])
    with patch.object(retrieval, "log_query") as log_query, \
         patch.object(retrieval, "vector_search", search), patch.object(retrieval, "hybrid_search", search):  # rag/search use vector, the agent tool hybrid
        c = TestClient(app)
        c.log_query, c.search, c.redis = log_query, search, redis
        c.meter = UsageMeter(client=redis, budget=50_000)
        yield c
    app.dependency_overrides.clear()


# ---- mode "rag" (Phase 8 behaviour, new response fields)
def test_rag_mode_is_the_default_and_returns_answer_citations_route_and_usage(route):
    llm.set_llm(FakeLLM("Use rhythm [1]. Pause more [2]. Invented [9]."))
    r = route.post("/api/ask", json={"question": Q})
    assert r.status_code == 200
    body = r.json()
    assert (body["mode"], body["route"], body["route_reason"]) == ("rag", "rag", "requested")
    assert body["answer"] == "Use rhythm [1]. Pause more [2]. Invented."
    assert body["citations"][0] == {"n": 1, "video_id": VID, "title": "How to Speak", "start_sec": 283.0, "end_sec": 313.0}
    assert [c["n"] for c in body["citations"]] == [1, 2]
    assert body["trace"] == [] and body["usage"] == {"llm_calls": 1, "tokens": 2}
    assert set(body) == {"answer", "citations", "mode", "route", "route_reason", "trace", "usage"}


def test_the_query_is_logged_with_mode_ask(route):
    llm.set_llm(FakeLLM("ok [1]"))
    route.post("/api/ask", json={"question": "  How   do I sound   sure? "})
    (_, user_id, question, mode), _ = route.log_query.call_args
    assert mode == "ask" and question == "How do I sound sure?" and user_id == USER


def test_without_matching_context_the_model_is_not_called(route):
    route.search.return_value = []
    fake = FakeLLM("never")
    llm.set_llm(fake)
    r = route.post("/api/ask", json={"question": "Airspeed of a swallow?"})
    assert r.status_code == 200 and r.json()["answer"] == rag.NO_CONTEXT_ANSWER and r.json()["citations"] == []
    assert fake.calls == [] and r.json()["usage"] == {"llm_calls": 0, "tokens": 0}


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "hi"}, {"question": "   "}, {"question": "x" * 1001},
                                     {"question": 123}, {"q": Q}, {"question": Q, "mode": "magic"}, {"question": Q, "session_id": "abc"},
                                     {"question": Q, "session_id": "has spaces in it"}, {"question": Q, "session_id": "x" * 65}])
def test_ask_validates_its_input(route, payload):
    llm.set_llm(FakeLLM())
    assert route.post("/api/ask", json=payload).status_code == 422


@pytest.mark.parametrize("error", [LLMUnavailable("api down key=SECRET"), LLMNotConfigured("no key"), LLMRejected("404 model SECRET"),
                                   LLMEmptyResponse("finish_reason=SAFETY")])
def test_llm_failures_become_503_with_a_safe_message(route, error):
    llm.set_llm(FakeLLM(error))
    r = route.post("/api/ask", json={"question": Q})
    assert r.status_code == 503 and r.json() == {"detail": error.user_message} and "SECRET" not in r.text


def test_ask_requires_login():
    app.dependency_overrides.clear()
    assert TestClient(app).post("/api/ask", json={"question": Q}).status_code == 401
    assert TestClient(app).get("/api/ask/stream", params={"q": Q}).status_code == 401
    assert TestClient(app).delete(f"/api/ask/session/{SID}").status_code == 401


# ---- mode "agent"
def agent_script():
    return ScriptedLLM([
        tools_turn(call("search_transcripts", query="sounding sure")),
        lambda msgs: text_turn(f"Slow down and pause [{VID}@283]. Vary pitch [{VID}@60]. Made up [{VID}@9999]."),
    ])


def test_agent_mode_runs_tools_and_returns_the_trace_and_checked_citations(route):
    llm.set_llm(agent_script())
    r = route.post("/api/ask", json={"question": Q, "mode": "agent"})
    assert r.status_code == 200
    body = r.json()
    assert (body["mode"], body["route"]) == ("agent", "agent")
    assert body["answer"] == "Slow down and pause [1]. Vary pitch [2]. Made up."
    assert [(c["n"], c["start_sec"], c["title"]) for c in body["citations"]] == [(1, 283.0, "How to Speak"), (2, 60.0, "How to Speak")]
    (step,) = body["trace"]
    assert (step["step"], step["tool"], step["summary"], step["error"]) == (1, "search_transcripts", "Found 2 clips", False)
    assert step["label"] == "Searching “sounding sure”" and step["args"] == {"query": "sounding sure"}
    assert body["usage"] == {"llm_calls": 2, "tokens": 200}


def test_the_per_request_llm_call_cap_ends_a_runaway_agent_gracefully(route, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_MAX_LLM_CALLS", 2)
    llm.set_llm(ScriptedLLM([tools_turn(call("search_transcripts", query="again"))] * 5))
    r = route.post("/api/ask", json={"question": Q, "mode": "agent"})
    assert r.status_code == 200  # not a 500: the evidence gathered so far is returned
    assert r.json()["usage"]["llm_calls"] == 2
    assert r.json()["answer"].startswith("I could not finish my research") and r.json()["citations"]


# ---- mode "auto"
def test_auto_sends_keyword_lookups_to_search_without_calling_the_model(route):
    fake = ScriptedLLM(reply="never used")
    llm.set_llm(fake)
    body = route.post("/api/ask", json={"question": "public speaking", "mode": "auto"}).json()
    assert (body["mode"], body["route"], body["route_reason"]) == ("auto", "search", "short keyword lookup")
    assert body["answer"].startswith("Top matches:\n[1] How to Speak at 04:43")
    assert [c["n"] for c in body["citations"]] == [1, 2] and body["usage"]["llm_calls"] == 0
    assert fake.calls == [] and fake.turn_calls == []
    assert route.search.call_args.kwargs["highlight"] is True and route.search.call_args.kwargs["k"] == 5


def test_auto_sends_single_questions_to_rag_and_comparisons_to_the_agent(route):
    llm.set_llm(ScriptedLLM([text_turn("Both talk about pauses.")], reply="A single fact [1]."))
    rag_body = route.post("/api/ask", json={"question": "How do rookie speakers sound?", "mode": "auto"}).json()
    assert (rag_body["route"], rag_body["answer"]) == ("rag", "A single fact [1].")
    agent_body = route.post("/api/ask", json={"question": "Compare what video A and video B say about pauses", "mode": "auto"}).json()
    assert (agent_body["route"], agent_body["route_reason"], agent_body["answer"]) == ("agent", "comparison", "Both talk about pauses.")


def test_auto_asks_the_model_only_for_ambiguous_input_and_that_call_counts(route):
    fake = ScriptedLLM([text_turn("Researched answer.")], reply="agent")
    llm.set_llm(fake)
    body = route.post("/api/ask", json={"question": "tips for a nervous speaker", "mode": "auto"}).json()
    assert (body["route"], body["route_reason"]) == ("agent", "ambiguous, classified by the model")
    assert len(fake.calls) == 1 and len(fake.turn_calls) == 1
    assert body["usage"]["llm_calls"] == 2  # classification + agent step, both under the same cap


# ---- conversation memory
def test_follow_ups_see_the_previous_turns_of_the_same_session_only(route):
    fake = ScriptedLLM([tools_turn(call("search_transcripts", query="pauses")), text_turn(f"Pause between ideas [{VID}@283]."),
                        text_turn("The second video says more."), text_turn("A different chat has no history."),
                        text_turn("No session id, no history.")])
    llm.set_llm(fake)
    first = route.post("/api/ask", json={"question": Q, "mode": "agent", "session_id": SID}).json()
    assert first["answer"] == "Pause between ideas [1]."
    route.post("/api/ask", json={"question": "and body language?", "mode": "auto", "session_id": SID})  # auto + history -> agent
    second_messages = fake.turn_calls[2]["messages"]
    assert [type(m).__name__ for m in second_messages] == ["UserMsg", "ModelMsg", "UserMsg"]
    assert second_messages[0].text == Q
    assert second_messages[1].text == "Pause between ideas (How to Speak @ 04:43)."  # [n] became (title @ mm:ss): n means nothing later
    assert second_messages[2].text == "and body language?"

    route.post("/api/ask", json={"question": Q, "mode": "agent", "session_id": "another-chat-9"})
    assert len(fake.turn_calls[3]["messages"]) == 1
    route.post("/api/ask", json={"question": Q, "mode": "agent"})  # no session id: nothing stored, nothing read
    assert len(fake.turn_calls[4]["messages"]) == 1


def test_memory_keeps_three_turns_and_can_be_cleared(route):
    fake = ScriptedLLM([text_turn(f"Answer {i}.") for i in range(6)])
    llm.set_llm(fake)
    for i in range(5):
        route.post("/api/ask", json={"question": f"Question number {i}", "mode": "agent", "session_id": SID})
    sent = fake.turn_calls[4]["messages"]
    assert [m.text for m in sent if type(m).__name__ == "UserMsg"] == ["Question number 1", "Question number 2", "Question number 3", "Question number 4"]
    assert route.delete(f"/api/ask/session/{SID}").status_code == 204
    route.post("/api/ask", json={"question": "fresh start please", "mode": "agent", "session_id": SID})
    assert len(fake.turn_calls[5]["messages"]) == 1
    assert route.delete("/api/ask/session/short").status_code == 422


# ---- daily token budget
def test_a_user_over_their_daily_budget_gets_a_429_with_retry_after_and_nothing_runs(route):
    route.meter.add(USER, 50_000)
    fake = ScriptedLLM([text_turn("never")], reply="never")
    llm.set_llm(fake)
    for payload in ({"question": Q}, {"question": Q, "mode": "agent"}, {"question": "public speaking", "mode": "auto"}):
        r = route.post("/api/ask", json=payload)
        assert r.status_code == 429, payload
        assert "midnight UTC" in r.json()["detail"] and 1 <= int(r.headers["Retry-After"]) <= DAY_SECONDS
    assert fake.calls == [] and fake.turn_calls == []
    route.search.assert_not_called()


def test_tokens_used_are_added_to_the_users_daily_total(route):
    llm.set_llm(agent_script())
    usage = route.post("/api/ask", json={"question": Q, "mode": "agent"}).json()["usage"]
    assert usage["tokens"] == 200 and route.meter.used(USER) == 200
    llm.set_llm(agent_script())
    route.post("/api/ask", json={"question": Q, "mode": "agent"})
    assert route.meter.used(USER) == 400  # accumulates across requests


# ---- streaming (Server-Sent Events)
def test_the_stream_emits_route_steps_then_the_answer_then_done(route):
    llm.set_llm(agent_script())
    r = route.get("/api/ask/stream", params={"q": Q, "mode": "agent"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in r.headers["cache-control"] and r.headers["x-accel-buffering"] == "no"
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["route", "step_start", "step_result", "answer", "done"]
    by = dict(events)
    assert by["route"] == {"type": "route", "route": "agent", "requested": "agent", "reason": "requested"}
    assert by["step_start"]["label"] == "Searching “sounding sure”" and by["step_start"]["tool"] == "search_transcripts"
    assert by["step_result"]["summary"] == "Found 2 clips" and by["step_result"]["error"] is False
    final = by["answer"]
    assert final["answer"] == "Slow down and pause [1]. Vary pitch [2]. Made up." and final["route"] == "agent"
    assert final["trace"][0]["summary"] == "Found 2 clips" and final["usage"]["llm_calls"] == 2


def test_the_stream_defaults_to_auto_and_search_needs_no_model(route):
    r = route.get("/api/ask/stream", params={"q": "public speaking"})
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["route", "answer", "done"]
    assert events[0][1]["route"] == "search" and events[0][1]["requested"] == "auto"
    assert events[1][1]["citations"][0]["n"] == 1


def test_the_stream_logs_the_query_and_uses_memory(route):
    fake = ScriptedLLM([text_turn("First answer."), text_turn("Second answer.")])
    llm.set_llm(fake)
    route.get("/api/ask/stream", params={"q": Q, "mode": "agent", "session_id": SID})
    route.get("/api/ask/stream", params={"q": "and body language?", "session_id": SID})  # auto + history -> agent
    assert route.log_query.call_count == 2 and route.log_query.call_args.args[3] == "ask"
    assert len(fake.turn_calls[1]["messages"]) == 3


def test_stream_errors_arrive_as_events_with_safe_messages(route):
    llm.set_llm(FakeLLM(LLMUnavailable("api down key=SECRET")))
    events = parse_sse(route.get("/api/ask/stream", params={"q": Q, "mode": "rag"}).text)
    names = [n for n, _ in events]
    assert names == ["route", "error", "done"]
    err = dict(events)["error"]
    assert err["status"] == 503 and err["message"] == LLMUnavailable("x").user_message and "SECRET" not in json.dumps(events)

    llm.set_llm(ScriptedLLM([RuntimeError("secret internal detail at 10.0.0.1")]))
    events = parse_sse(route.get("/api/ask/stream", params={"q": Q, "mode": "agent"}).text)
    err = dict(events)["error"]
    assert err["status"] == 500 and err["message"] == ask_routes.GENERIC_ERROR and "10.0.0.1" not in json.dumps(events)
    assert events[-1][0] == "done"


def test_the_stream_refuses_with_a_real_http_429_when_the_daily_budget_is_gone(route):
    route.meter.add(USER, 50_000)
    llm.set_llm(ScriptedLLM([text_turn("never")]))
    r = route.get("/api/ask/stream", params={"q": Q, "mode": "agent"})
    assert r.status_code == 429 and "midnight UTC" in r.json()["detail"] and int(r.headers["Retry-After"]) >= 1
    assert r.headers["content-type"].startswith("application/json")  # an ordinary error response, not a stream


@pytest.mark.parametrize("params", [{}, {"q": "hi"}, {"q": "   "}, {"q": Q, "mode": "magic"}, {"q": Q, "session_id": "no"}, {"q": "x" * 1001}])
def test_the_stream_validates_its_input(route, params):
    assert route.get("/api/ask/stream", params=params).status_code == 422


def test_closing_the_stream_cancels_the_agent_before_its_next_model_call(route):
    """A client that disconnects must not keep burning LLM calls."""
    gate = threading.Event()
    seen = []

    def first_turn(msgs):
        seen.append("first")
        gate.wait(5)  # the model is "thinking" while the client leaves
        return tools_turn(call("search_transcripts", query="sounding sure"))

    fake = ScriptedLLM([first_turn, text_turn("this second call must never happen")])
    llm.set_llm(fake)
    stream = ask_routes.event_stream(MagicMock, USER, Q, "agent", None, UsageMeter(client=FakeRedis()), ChatMemory(client=FakeRedis()), 5)
    assert next(stream).startswith("event: route")
    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        time.sleep(0.01)
    stream.close()  # what Starlette does when the client disconnects
    gate.set()
    for t in [t for t in threading.enumerate() if t.name == "ask-stream"]:
        t.join(5)
    assert len(fake.turn_calls) == 1  # the loop saw the cancel flag and never asked the model again


def test_an_idle_stream_sends_keep_alive_comments(route):
    gate = threading.Event()
    fake = ScriptedLLM([lambda msgs: (gate.wait(5), text_turn("done"))[1]])
    llm.set_llm(fake)
    stream = ask_routes.event_stream(MagicMock, USER, Q, "agent", None, UsageMeter(client=FakeRedis()), ChatMemory(client=FakeRedis()), 0.05)
    assert next(stream).startswith("event: route")
    assert next(stream) == ": keep-alive\n\n"  # nothing happened for 50 ms
    gate.set()
    rest = list(stream)
    assert [parse_sse(chunk)[0][0] for chunk in rest if not chunk.startswith(":")] == ["answer", "done"]


# ---------------------------------------------------------------- integration: real DB + Redis
DIM = settings.EMBED_DIM
QUERY = [1.0] + [0.0] * (DIM - 1)


def vec(score):
    return [score, math.sqrt(1 - score * score)] + [0.0] * (DIM - 2)


@pytest.fixture
def seeded(api, db_session, redis_test):
    v = Video(id=uuid.uuid4(), title="Public speaking", storage_key="raw/a.mp4", status="ready", duration_sec=300)
    db_session.add(v)
    db_session.flush()
    db_session.add_all([
        Chunk(id=uuid.uuid4(), video_id=v.id, idx=0, start_sec=0, end_sec=30, text="Slow down and pause between ideas.", embedding=vec(0.9)),
        Chunk(id=uuid.uuid4(), video_id=v.id, idx=1, start_sec=100, end_sec=130, text="Vary your pitch to keep attention.", embedding=vec(0.7)),
        Chunk(id=uuid.uuid4(), video_id=v.id, idx=2, start_sec=200, end_sec=230, text="Unrelated tangent.", embedding=vec(0.1)),
    ])
    db_session.commit()
    api.video_id, api.redis = str(v.id), redis_test
    with patch.object(retrieval, "embed_query", return_value=QUERY):
        yield api


def login(api, email="ann@example.com"):
    api.post("/auth/register", json={"email": email, "password": "Passw0rd123"})
    token = api.post("/auth/login", json={"email": email, "password": "Passw0rd123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def real_agent(video_id):
    """Searches and lists in parallel, reads a window around the best clip, then answers citing real refs."""
    def read_window(msgs):
        clips = next(r.content for r in next(m for m in reversed(msgs) if hasattr(m, "responses")).responses if r.name == "search_transcripts")["clips"]
        return tools_turn(call("get_transcript_window", video_id=clips[0]["video_id"], start_sec=clips[0]["start_sec"], span_sec=120))

    return ScriptedLLM([
        tools_turn(call("search_transcripts", query="pacing and pauses", k=3), call("list_videos", title_contains="speaking")),
        read_window,
        lambda msgs: text_turn(f"Pause between ideas [{video_id}@0] and vary pitch [{video_id}@100]. Unseen [{video_id}@200]."),
    ])


def test_rag_end_to_end_against_pgvector(seeded):
    fake = FakeLLM("Pause between ideas [1] and vary pitch [2]. Made up [8].")
    llm.set_llm(fake)
    r = seeded.post("/api/ask", json={"question": "How should I pace myself?"}, headers=login(seeded))
    assert r.status_code == 200
    assert r.json()["answer"] == "Pause between ideas [1] and vary pitch [2]. Made up."
    assert [(c["n"], c["video_id"], c["start_sec"]) for c in r.json()["citations"]] == [(1, seeded.video_id, 0.0), (2, seeded.video_id, 100.0)]
    assert "[1] (Public speaking @ 00:00) Slow down and pause between ideas." in fake.calls[0]["prompt"]
    assert "Unrelated tangent" not in fake.calls[0]["prompt"]  # below MIN_SCORE


def test_the_agent_uses_real_services_in_parallel_and_cites_only_what_it_was_shown(seeded):
    fake = real_agent(seeded.video_id)
    llm.set_llm(fake)
    r = seeded.post("/api/ask", json={"question": "How should I pace myself?", "mode": "agent"}, headers=login(seeded))
    assert r.status_code == 200, r.text
    body = r.json()

    assert [(s["step"], s["index"], s["tool"], s["summary"]) for s in body["trace"]] == [
        (1, 0, "search_transcripts", "Found 2 clips"), (1, 1, "list_videos", "Found 1 video"),
        (2, 0, "get_transcript_window", "Read 00:00–02:00 of “Public speaking”")]
    assert [s["error"] for s in body["trace"]] == [False] * 3
    # the second model call saw both results of step 1 in ONE tool message, searched through real SQL
    first_results = {r.name: r.content for r in fake.turn_calls[1]["messages"][-1].responses}
    assert [c["start_sec"] for c in first_results["search_transcripts"]["clips"]] == [0.0, 100.0]
    assert first_results["list_videos"]["videos"][0]["title"] == "Public speaking"
    window = {r.name: r.content for r in fake.turn_calls[2]["messages"][-1].responses}["get_transcript_window"]
    assert window["segments"] and window["segments"][0]["text"].startswith("<transcript_excerpt>Slow down")

    # citations: refs the tools returned are kept (as [n]); [video@200] (the "unrelated" chunk was never shown) is removed
    assert body["answer"] == "Pause between ideas [1] and vary pitch [2]. Unseen."
    assert [(c["n"], c["start_sec"]) for c in body["citations"]] == [(1, 0.0), (2, 100.0)]


def test_tokens_memory_and_the_budget_use_real_redis(seeded):
    headers = login(seeded)
    fake = ScriptedLLM([text_turn("First."), text_turn("Second.")])
    llm.set_llm(fake)
    r = seeded.post("/api/ask", json={"question": "How should I pace myself?", "mode": "agent", "session_id": SID}, headers=headers)
    assert r.json()["usage"]["tokens"] == 100
    key = next(k.decode() for k in seeded.redis.keys("tokens:*"))
    assert int(seeded.redis.get(key)) == 100 and 0 < seeded.redis.ttl(key) <= 2 * DAY_SECONDS
    seeded.post("/api/ask", json={"question": "and the pitch?", "mode": "auto", "session_id": SID}, headers=headers)
    assert [type(m).__name__ for m in fake.turn_calls[1]["messages"]] == ["UserMsg", "ModelMsg", "UserMsg"]  # history came from Redis
    assert seeded.redis.llen(f"chat:{next(k.decode().split(':')[1] for k in seeded.redis.keys('tokens:*'))}:{SID}") == 2

    seeded.redis.set(key, settings.DAILY_TOKEN_BUDGET)  # the day's budget is used up
    blocked = seeded.post("/api/ask", json={"question": "one more please", "mode": "agent"}, headers=headers)
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) >= 1
    assert len(fake.turn_calls) == 2  # nothing ran


def test_the_stream_end_to_end_with_real_services(seeded):
    llm.set_llm(real_agent(seeded.video_id))
    r = seeded.get("/api/ask/stream", params={"q": "How should I pace myself?", "mode": "agent"}, headers=login(seeded))
    assert r.status_code == 200
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "route" and names[-2:] == ["answer", "done"]
    assert names.count("step_start") == 3 and names.count("step_result") == 3
    labels = [d["label"] for n, d in events if n == "step_start"]
    assert labels[0] == "Searching “pacing and pauses”" and "Listing videos" in labels[1] and labels[2].startswith("Reading 00:00")
    assert dict(events)["answer"]["citations"][0]["start_sec"] == 0.0
    modes = seeded.get("/health")  # noqa: F841
    from app.core.db import engine
    with engine.connect() as conn:
        assert conn.execute(text("SELECT mode FROM search_logs")).scalars().all() == ["ask"]


def test_ask_is_rate_limited_to_10_per_minute_per_user(seeded):
    llm.set_llm(FakeLLM("ok [1]"))
    alice, bob = login(seeded, "alice@example.com"), login(seeded, "bob@example.com")
    body = {"question": "How should I pace myself?"}
    assert [seeded.post("/api/ask", json=body, headers=alice).status_code for _ in range(10)] == [200] * 10
    blocked = seeded.post("/api/ask", json=body, headers=alice)
    assert blocked.status_code == 429 and 1 <= int(blocked.headers["Retry-After"]) <= 60
    assert seeded.get("/api/ask/stream", params={"q": body["question"], "mode": "rag"}, headers=alice).status_code == 429  # same bucket
    assert seeded.post("/api/ask", json=body, headers=bob).status_code == 200
    assert len(llm.get_llm().calls) == 11


def test_ask_with_nothing_relevant_in_the_database_skips_the_model(seeded):
    fake = FakeLLM("never")
    llm.set_llm(fake)
    with patch.object(retrieval, "embed_query", return_value=[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)):
        r = seeded.post("/api/ask", json={"question": "Something unrelated entirely"}, headers=login(seeded))
    assert r.json()["answer"] == rag.NO_CONTEXT_ANSWER and r.json()["citations"] == [] and fake.calls == []


def test_orchestrator_module_is_what_both_endpoints_use():
    assert ask_routes.orchestrator is orchestrator
