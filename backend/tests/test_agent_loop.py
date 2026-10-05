"""The agent loop, driven by a scripted fake LLM (a planned sequence of model turns) and fake services."""
import threading
from unittest.mock import MagicMock

import pytest

from app.agent import loop
from app.agent.limits import CallBudget, CallCapExceeded, Cancelled, MeteredLLM
from app.agent.loop import run_agent
from app.agent.prompts import FORCE_FINAL_MESSAGE, SYSTEM_PROMPT
from app.agent.tools import TOOL_SPECS
from app.services import catalog, retrieval, windowing
from app.services.llm import ModelMsg, ToolMsg, UserMsg
from app.services.usage import DailyBudgetExceeded, UsageMeter
from tests.fakes import FakeRedis, ScriptedLLM, call, hit, text_turn, tools_turn

VID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
VID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
factory = MagicMock


@pytest.fixture
def library(monkeypatch):
    """Fake retrieval: 'rookie' finds clips in video A, 'natural' in video B, anything else nothing."""
    calls = []

    def fake_search(db, query, k=10, **kw):
        calls.append(query)
        if "rookie" in query:
            return [hit(1, video_id=VID_A, title="Video A", start=100.0, text="Rookies speak flat."),
                    hit(2, video_id=VID_A, title="Video A", start=130.0, text="Rookies avoid gestures.")]
        if "natural" in query:
            return [hit(3, video_id=VID_B, title="Video B", start=40.0, text="Naturals vary pitch.")]
        return []

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    monkeypatch.setattr(windowing, "get_window", lambda db, vid, start, end: {
        "video_id": str(vid), "title": "Video A", "segments": [{"start_sec": start, "end_sec": end, "text": "Window text."}]})
    monkeypatch.setattr(catalog, "list_ready_videos", lambda db, **kw: [
        {"video_id": VID_A, "title": "Video A", "duration_sec": 600, "uploaded": "2026-03-01"}])
    return calls


def tool_results(messages):
    """The dicts of the last ToolMsg: what the model was shown."""
    return [r.content for r in next(m for m in reversed(messages) if isinstance(m, ToolMsg)).responses]


# ---- a normal multi-step run
def test_multi_step_run_searches_twice_then_answers_with_checked_citations(library):
    llm = ScriptedLLM([
        tools_turn(call("search_transcripts", query="how do rookie speakers sound")),
        tools_turn(call("search_transcripts", query="how do natural speakers sound")),
        lambda msgs: text_turn(f"Rookies sound flat [{VID_A}@100]. Naturals vary pitch [{VID_B}@40]."),
    ])
    result = run_agent(factory, "Compare rookie and natural speakers", llm=llm)

    assert library == ["how do rookie speakers sound", "how do natural speakers sound"]
    assert result.steps == 2 and result.llm_calls == 3 and result.stopped == "answer"
    assert result.answer == "Rookies sound flat [1]. Naturals vary pitch [2]."
    assert [(c.n, c.video_id, c.title, c.start_sec, c.end_sec) for c in result.citations] == [
        (1, VID_A, "Video A", 100.0, 130.0), (2, VID_B, "Video B", 40.0, 70.0)]
    assert [(t.step, t.index, t.tool, t.summary, t.error) for t in result.trace] == [
        (1, 0, "search_transcripts", "Found 2 clips", False), (2, 0, "search_transcripts", "Found 1 clip", False)]
    assert result.trace[0].label == "Searching “how do rookie speakers sound”"
    assert all(t.latency_ms >= 0 for t in result.trace)


def test_what_the_model_is_sent_on_each_step(library):
    llm = ScriptedLLM([tools_turn(call("search_transcripts", query="rookie")), text_turn("Done.")])
    run_agent(factory, "Tell me about rookies", llm=llm)

    first, second = llm.turn_calls
    assert first["system"] == SYSTEM_PROMPT and first["allow_tools"] is True
    assert first["messages"] == [UserMsg("Tell me about rookies")]
    assert [t.name for t in first["tools"]] == [s.name for s in TOOL_SPECS]
    # step 2 sees: the question, its own tool call, and the tool result (in that order)
    kinds = [type(m).__name__ for m in second["messages"]]
    assert kinds == ["UserMsg", "ModelMsg", "ToolMsg"]
    assert second["messages"][1].tool_calls[0].name == "search_transcripts"
    assert second["messages"][2].responses[0].content["count"] == 2
    assert second["messages"][2].responses[0].name == "search_transcripts"


def test_an_immediate_answer_uses_one_call_and_no_tools(library):
    result = run_agent(factory, "Hi there", llm=ScriptedLLM([text_turn("Hello! Ask me about the videos.")]))
    assert (result.steps, result.llm_calls, result.trace, result.citations) == (0, 1, [], [])
    assert result.answer == "Hello! Ask me about the videos."


def test_text_that_comes_with_tool_calls_is_not_the_answer(library):
    llm = ScriptedLLM([tools_turn(call("search_transcripts", query="rookie"), text="Let me look that up."), text_turn("Final.")])
    result = run_agent(factory, "q", llm=llm)
    assert result.answer == "Final."
    assert llm.turn_calls[1]["messages"][1].text == "Let me look that up."  # kept in the history, never shown as the answer


# ---- several calls in one step run in parallel
def test_calls_in_one_step_really_run_at_the_same_time(monkeypatch):
    barrier = threading.Barrier(2, timeout=3)  # both searches must be inside the tool at once, or this raises
    threads = set()

    def fake_search(db, query, k=10, **kw):
        threads.add(threading.get_ident())
        barrier.wait()
        return [hit(1, video_id=VID_A, start=100.0)] if "A" in query else [hit(2, video_id=VID_B, start=40.0)]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    llm = ScriptedLLM([
        tools_turn(call("search_transcripts", query="topic in A"), call("search_transcripts", query="topic in B")),
        text_turn(f"A says this [{VID_A}@100]; B says that [{VID_B}@40]."),
    ])
    result = run_agent(factory, "Compare A and B", llm=llm)

    assert len(threads) == 2  # two worker threads
    assert result.steps == 1 and result.llm_calls == 2
    assert [(t.step, t.index) for t in result.trace] == [(1, 0), (1, 1)]
    tool_msg = llm.turn_calls[1]["messages"][-1]
    assert isinstance(tool_msg, ToolMsg) and len(tool_msg.responses) == 2  # ONE message carrying both results
    assert [r.content["query"] for r in tool_msg.responses] == ["topic in A", "topic in B"]  # same order as the calls
    assert [c.video_id for c in result.citations] == [VID_A, VID_B]


def test_a_single_call_does_not_need_a_thread(library):
    main = threading.get_ident()
    seen = []
    orig = loop.execute_tool

    def spy(*a, **k):
        seen.append(threading.get_ident())
        return orig(*a, **k)

    loop.execute_tool = spy
    try:
        run_agent(factory, "q", llm=ScriptedLLM([tools_turn(call("search_transcripts", query="rookie")), text_turn("ok")]))
    finally:
        loop.execute_tool = orig
    assert seen == [main]


def test_more_calls_than_the_limit_get_an_error_back_instead_of_running(library):
    many = [call("search_transcripts", query=f"rookie {i}") for i in range(6)]
    llm = ScriptedLLM([tools_turn(*many), text_turn("done")])
    result = run_agent(factory, "q", llm=llm)
    contents = [r.content for r in llm.turn_calls[1]["messages"][-1].responses]
    assert len(contents) == 6  # every call is answered (the API requires it)
    assert all("error" not in c for c in contents[:4])
    assert all("Too many tool calls" in c["error"] for c in contents[4:])
    assert [t.error for t in result.trace] == [False] * 4 + [True] * 2
    assert len(library) == 4  # only 4 searches actually ran


# ---- bad arguments: the model recovers
def test_bad_arguments_are_reported_to_the_model_which_then_fixes_them(library):
    llm = ScriptedLLM([
        tools_turn(call("search_transcripts", query="x")),  # too short
        tools_turn(call("search_transcripts", query="rookie speakers", k="lots")),  # wrong type
        tools_turn(call("search_transcripts", query="rookie speakers")),  # fixed
        text_turn(f"Rookies speak flat [{VID_A}@100]."),
    ])
    result = run_agent(factory, "q", llm=llm)

    first_error = tool_results(llm.turn_calls[1]["messages"])[0]["error"]
    second_error = tool_results(llm.turn_calls[2]["messages"])[0]["error"]
    assert "query" in first_error and "call the tool again" in first_error
    assert "k:" in second_error
    assert [t.error for t in result.trace] == [True, True, False]
    assert result.trace[0].summary.startswith("Error: Invalid arguments")
    assert result.answer == "Rookies speak flat [1]." and result.stopped == "answer"
    assert library == ["rookie speakers"]  # the invalid calls never reached the service


def test_unknown_tools_and_crashing_tools_do_not_break_the_loop(monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    llm = ScriptedLLM([
        tools_turn(call("web_search", query="anything")),
        tools_turn(call("search_transcripts", query="rookie")),
        text_turn("I could not search the videos right now."),
    ])
    result = run_agent(factory, "q", llm=llm)
    assert "Unknown tool" in tool_results(llm.turn_calls[1]["messages"])[0]["error"]
    assert "failed unexpectedly" in tool_results(llm.turn_calls[2]["messages"])[0]["error"]
    assert result.answer == "I could not search the videos right now." and result.citations == []


# ---- max steps
def test_after_max_steps_the_model_is_forced_to_answer_with_what_it_has(library):
    keep_searching = [tools_turn(call("search_transcripts", query="rookie")) for _ in range(3)]
    llm = ScriptedLLM([*keep_searching, text_turn(f"Best I found: rookies speak flat [{VID_A}@100].")])
    result = run_agent(factory, "q", llm=llm, max_steps=3)

    assert result.stopped == "max_steps" and result.steps == 3
    assert result.llm_calls == 4  # 3 steps + 1 forced final: never more
    assert len(result.trace) == 3
    last = llm.turn_calls[3]
    assert last["allow_tools"] is False  # the model is not allowed to call tools on the last turn
    assert last["messages"][-1] == UserMsg(FORCE_FINAL_MESSAGE)
    assert "ONLY the evidence already gathered" in FORCE_FINAL_MESSAGE
    assert all(c["allow_tools"] for c in llm.turn_calls[:3])
    assert result.answer == "Best I found: rookies speak flat [1]." and [c.video_id for c in result.citations] == [VID_A]


def test_if_the_forced_final_turn_is_empty_the_answer_falls_back_to_the_evidence(library):
    llm = ScriptedLLM([tools_turn(call("search_transcripts", query="rookie")), tools_turn(call("search_transcripts", query="natural")),
                       tools_turn(call("search_transcripts", query="rookie"))] + [text_turn("   ")])
    result = run_agent(factory, "q", llm=llm, max_steps=3)
    assert result.stopped == "max_steps"
    assert result.answer.startswith("I could not finish my research")
    assert len(result.citations) == 3 and "[1]" in result.answer  # the evidence it did gather, as real citations


def test_default_max_steps_comes_from_settings(library, monkeypatch):
    monkeypatch.setattr(loop.settings, "AGENT_MAX_STEPS", 2)
    llm = ScriptedLLM([tools_turn(call("list_videos")), tools_turn(call("list_videos")), text_turn("ok")])
    assert run_agent(factory, "q", llm=llm).llm_calls == 3


# ---- guardrails: call cap, daily budget, cancellation
def test_the_call_cap_ends_the_run_gracefully(library):
    inner = ScriptedLLM([tools_turn(call("search_transcripts", query="rookie")), tools_turn(call("search_transcripts", query="natural"))])
    metered = MeteredLLM(inner, CallBudget(2))
    result = run_agent(factory, "q", llm=metered)  # 3rd model call would exceed the cap of 2
    assert result.stopped == "budget" and result.llm_calls == 2
    assert result.answer.startswith("I could not finish my research")
    assert {c.video_id for c in result.citations} == {VID_A, VID_B}
    assert len(inner.turn_calls) == 2  # the capped call never reached the model


def test_running_out_of_daily_tokens_mid_run_ends_it_gracefully(library):
    meter = UsageMeter(client=FakeRedis(), budget=150)
    inner = ScriptedLLM([tools_turn(call("search_transcripts", query="rookie"), total_tokens=200), text_turn("never reached")])
    metered = MeteredLLM(inner, CallBudget(8), meter=meter, user_id="u1")
    result = run_agent(factory, "q", llm=metered)
    assert result.stopped == "budget" and result.llm_calls == 1
    assert len(inner.turn_calls) == 1 and result.answer.startswith("I could not finish")


def test_cancellation_stops_before_the_next_model_call(library):
    stop = threading.Event()
    inner = ScriptedLLM([lambda msgs: (stop.set(), tools_turn(call("search_transcripts", query="rookie")))[1], text_turn("never")])
    with pytest.raises(Cancelled):
        run_agent(factory, "q", llm=inner, should_stop=stop.is_set)
    assert len(inner.turn_calls) == 1


# ---- conversation history
def test_earlier_turns_are_sent_before_the_new_question(library):
    history = [UserMsg("What do rookies do?"), ModelMsg(text="They speak flat (Video A @ 01:40).")]
    llm = ScriptedLLM([text_turn("The second video says more.")])
    run_agent(factory, "what about the second video?", llm=llm, history=history)
    assert llm.turn_calls[0]["messages"] == [*history, UserMsg("what about the second video?")]


# ---- evidence and truncation interact: you can only cite what the model was shown
def test_a_clip_cut_by_the_token_budget_cannot_be_cited(monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [
        hit(i, video_id=VID_A, start=100.0 * i, text="w " * 300) for i in range(1, 6)])
    llm = ScriptedLLM([tools_turn(call("search_transcripts", query="long text")),
                       text_turn(f"First [{VID_A}@100] and last [{VID_A}@500].")])
    result = run_agent(factory, "q", llm=llm, tool_token_budget=300)
    shown = tool_results(llm.turn_calls[1]["messages"])[0]
    assert shown["truncated"] is True and len(shown["clips"]) < 5
    assert result.answer == "First [1] and last."  # clip 5 was dropped before the model saw it, so citing it is invalid
    assert [c.start_sec for c in result.citations] == [100.0]


# ---- events for streaming
def test_progress_events_are_emitted_in_order(library):
    events = []
    llm = ScriptedLLM([tools_turn(call("search_transcripts", query="rookie")), tools_turn(call("get_transcript_window", video_id=VID_A, start_sec=130)),
                       text_turn("ok")])
    run_agent(factory, "q", llm=llm, on_event=events.append)
    assert [(e["type"], e["step"], e["tool"]) for e in events] == [
        ("step_start", 1, "search_transcripts"), ("step_result", 1, "search_transcripts"),
        ("step_start", 2, "get_transcript_window"), ("step_result", 2, "get_transcript_window")]
    assert events[0]["label"] == "Searching “rookie”" and events[0]["args"] == {"query": "rookie"}
    assert events[1]["summary"] == "Found 2 clips" and events[1]["error"] is False
    assert events[3]["summary"] == "Read 02:10–03:10 of “Video A”"


def test_long_string_arguments_are_shortened_in_the_trace(library):
    long = "rookie " + "x" * 500
    result = run_agent(factory, "q", llm=ScriptedLLM([tools_turn(call("search_transcripts", query=long)), text_turn("ok")]))
    assert len(result.trace[0].args["query"]) <= 201 and result.trace[0].args["query"].endswith("…")


def test_a_final_answer_with_no_text_falls_back(library):
    result = run_agent(factory, "q", llm=ScriptedLLM([text_turn("")]))
    assert "couldn't find anything" in result.answer
