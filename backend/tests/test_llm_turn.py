"""GeminiLLM.generate_turn: the request it builds and how it reads the reply, with a fake SDK client."""
from types import SimpleNamespace

import pytest

from app.agent.tools import TOOL_SPECS
from app.services.llm import (GeminiLLM, LLMEmptyResponse, LLMUnavailable, ModelMsg, ToolCall, ToolMsg, ToolResponse, UserMsg)
from tests.test_llm import ApiError, FakeClient, make


def part(text=None, call=None, thought=False):
    fc = SimpleNamespace(name=call[0], args=call[1], id=call[2] if len(call) > 2 else None) if call else None
    return SimpleNamespace(text=text, function_call=fc, thought=thought)


def reply(*parts, usage=None):
    content = SimpleNamespace(role="model", parts=list(parts))
    usage = usage or SimpleNamespace(prompt_token_count=500, candidates_token_count=20, thoughts_token_count=30, total_token_count=550)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content, finish_reason="STOP")], usage_metadata=usage)


def turn(client, messages=None, **kw):
    return make(client).generate_turn(system="sys", messages=messages or [UserMsg("hi")], tools=kw.pop("tools", TOOL_SPECS), **kw)


# ---- the request
def test_tools_are_declared_with_their_schema_and_automatic_calling_is_off():
    client = FakeClient(reply(part(text="hello")))
    turn(client)
    cfg = client.calls[0]["config"]
    decls = cfg["tools"][0]["function_declarations"]
    assert [d["name"] for d in decls] == ["search_transcripts", "list_videos", "get_transcript_window"]
    assert decls[0]["parameters"]["required"] == ["query"] and decls[0]["description"]
    assert cfg["automatic_function_calling"] == {"disable": True}  # our loop runs the tools, not the SDK
    assert "tool_config" not in cfg
    assert cfg["system_instruction"] == "sys" and cfg["http_options"] == {"timeout": 5000} and cfg["max_output_tokens"] == 256


def test_a_forced_final_answer_keeps_tools_declared_but_forbids_calling_them():
    client = FakeClient(reply(part(text="final")))
    turn(client, allow_tools=False)
    cfg = client.calls[0]["config"]
    assert cfg["tool_config"] == {"function_calling_config": {"mode": "NONE"}}
    assert cfg["tools"]  # still declared: the history contains tool calls the API needs to understand


def test_without_tools_nothing_tool_related_is_sent():
    client = FakeClient(reply(part(text="plain")))
    turn(client, tools=None)
    assert not {"tools", "tool_config", "automatic_function_calling"} & set(client.calls[0]["config"])


def test_messages_are_converted_to_gemini_contents():
    raw = SimpleNamespace(role="model", parts=["opaque, with a thought signature"])
    msgs = [
        UserMsg("question"),
        ModelMsg(text="thinking out loud", tool_calls=(ToolCall("search_transcripts", {"query": "a"}, id="c1"), ToolCall("list_videos", {}))),
        ToolMsg((ToolResponse("search_transcripts", {"count": 1}, id="c1"), ToolResponse("list_videos", {"count": 0}))),
        ModelMsg(text="kept as is", raw=raw),
    ]
    client = FakeClient(reply(part(text="ok")))
    turn(client, msgs)
    contents = client.calls[0]["contents"]
    assert contents[0] == {"role": "user", "parts": [{"text": "question"}]}
    assert contents[1] == {"role": "model", "parts": [
        {"text": "thinking out loud"},
        {"function_call": {"name": "search_transcripts", "args": {"query": "a"}, "id": "c1"}},
        {"function_call": {"name": "list_videos", "args": {}}}]}
    assert contents[2] == {"role": "user", "parts": [  # ONE user turn with every result, in call order
        {"function_response": {"name": "search_transcripts", "response": {"count": 1}, "id": "c1"}},
        {"function_response": {"name": "list_videos", "response": {"count": 0}}}]}
    assert contents[3] is raw  # the model's own turn is echoed untouched (keeps thought signatures)


# ---- the reply
def test_function_calls_are_parsed_in_order_with_their_arguments_and_ids():
    client = FakeClient(reply(part(call=("search_transcripts", {"query": "a", "k": 5.0}, "id1")), part(call=("list_videos", {}))))
    t = turn(client)
    assert t.tool_calls == (ToolCall("search_transcripts", {"query": "a", "k": 5.0}, "id1"), ToolCall("list_videos", {}, None))
    assert t.text is None
    assert t.raw is not None


def test_text_is_returned_and_thinking_parts_are_ignored():
    client = FakeClient(reply(part(text="private reasoning", thought=True), part(text="The answer. "), part(text="More.")))
    t = turn(client)
    assert t.text == "The answer. More." and t.tool_calls == ()


def test_text_may_come_with_calls():
    t = turn(FakeClient(reply(part(text="Let me check."), part(call=("list_videos", {})))))
    assert t.text == "Let me check." and len(t.tool_calls) == 1


def test_raw_is_the_models_content_object_for_echoing_back():
    r = reply(part(call=("list_videos", {})))
    assert turn(FakeClient(r)).raw is r.candidates[0].content


def test_usage_is_reported_and_logged_like_generate(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="app.services.llm"):
        t = turn(FakeClient(reply(part(text="hello"))))
    assert (t.prompt_tokens, t.completion_tokens, t.thought_tokens, t.total_tokens, t.attempts, t.model) == (500, 20, 30, 550, 1, "m-1")
    assert "prompt_tokens=500" in caplog.text and "hello" not in caplog.text


def test_empty_replies_raise():
    for bad in (reply(), reply(part(text="  ")), SimpleNamespace(candidates=[], usage_metadata=None)):
        with pytest.raises(LLMEmptyResponse):
            turn(FakeClient(bad))


def test_turns_are_retried_on_overload_like_any_other_call():
    client = FakeClient(ApiError(503), ApiError(429), reply(part(text="ok")))
    t = turn(client)
    assert t.attempts == 3 and len(client.calls) == 3
    with pytest.raises(LLMUnavailable):
        turn(FakeClient(ApiError(503), ApiError(503), ApiError(503)))
