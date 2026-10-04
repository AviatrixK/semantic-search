"""The Gemini wrapper, tested with a fake client: no SDK call, no network, no API key."""
import logging
from types import SimpleNamespace

import pytest

from app.services import llm
from app.services.llm import (GeminiLLM, LLMEmptyResponse, LLMError, LLMNotConfigured, LLMRejected, LLMUnavailable,
                              backoff_delay, call_with_retries, is_retryable)
from tests.fakes import FakeLLM


class ApiError(Exception):
    """Looks like google.genai.errors.APIError: the HTTP status is in `.code`."""

    def __init__(self, code, message="boom"):
        super().__init__(message)
        self.code = code


def response(text="An answer [1].", **usage):
    usage = {"prompt_token_count": 120, "candidates_token_count": 30, "thoughts_token_count": 7, "total_token_count": 157, **usage}
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(**usage), candidates=[SimpleNamespace(finish_reason="STOP")])


class FakeClient:
    """client.models.generate_content(...) plays the scripted outcomes in order (an Exception is raised)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make(client, *, max_retries=2, api_key="key", sleeps=None):
    return GeminiLLM(model="m-1", api_key=api_key, timeout_sec=5, max_retries=max_retries, max_output_tokens=256, client=client,
                     sleep=(sleeps.append if sleeps is not None else lambda s: None), jitter=lambda: 1.0)


# ---- which errors are retried
@pytest.mark.parametrize("code,expected", [(429, True), (500, True), (502, True), (503, True), (504, True),
                                           (400, False), (401, False), (403, False), (404, False), (422, False)])
def test_retryable_by_http_status(code, expected):
    assert is_retryable(ApiError(code)) is expected


def test_network_trouble_is_retryable_other_errors_are_not():
    import httpx
    assert is_retryable(TimeoutError()) and is_retryable(ConnectionError())
    assert is_retryable(httpx.ReadTimeout("slow")) and is_retryable(httpx.ConnectError("refused"))
    assert not is_retryable(ValueError("bug")) and not is_retryable(KeyError("x"))


def test_backoff_doubles_is_capped_and_jittered_between_half_and_full():
    assert backoff_delay(0, 1.0, 10.0, 1.0) == 1.0 and backoff_delay(0, 1.0, 10.0, 0.0) == 0.5
    assert [backoff_delay(a, 1.0, 100.0, 1.0) for a in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert backoff_delay(10, 1.0, 10.0, 1.0) == 10.0  # capped


# ---- retry loop
def test_call_with_retries_returns_result_and_attempt_count():
    calls = iter([ApiError(503), ApiError(429), "ok"])
    sleeps = []

    def fn():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    result, attempts = call_with_retries(fn, max_retries=3, sleep=sleeps.append, jitter=lambda: 1.0)
    assert (result, attempts) == ("ok", 3)
    assert sleeps == [1.0, 2.0]  # exponential backoff between attempts


def test_call_with_retries_gives_up_after_max_retries_and_raises_the_last_error():
    n = 0

    def fn():
        nonlocal n
        n += 1
        raise ApiError(503, f"attempt {n}")

    with pytest.raises(ApiError, match="attempt 3"):
        call_with_retries(fn, max_retries=2, sleep=lambda s: None)
    assert n == 3  # 1 call + 2 retries


def test_call_with_retries_does_not_retry_client_errors():
    n = 0

    def fn():
        nonlocal n
        n += 1
        raise ApiError(400, "bad request")

    sleeps = []
    with pytest.raises(ApiError):
        call_with_retries(fn, max_retries=5, sleep=sleeps.append)
    assert n == 1 and sleeps == []


def test_zero_retries_means_a_single_attempt():
    n = 0

    def fn():
        nonlocal n
        n += 1
        raise ApiError(503)

    with pytest.raises(ApiError):
        call_with_retries(fn, max_retries=0)
    assert n == 1


# ---- GeminiLLM
def test_generate_sends_model_prompt_and_config_and_returns_text_and_usage():
    client = FakeClient(response("Use your voice [1]."))
    result = make(client).generate(system="be strict", prompt="the prompt")
    call = client.calls[0]
    assert call["model"] == "m-1" and call["contents"] == "the prompt"
    assert call["config"]["system_instruction"] == "be strict"
    assert call["config"]["temperature"] <= 0.3
    assert call["config"]["max_output_tokens"] == 256
    assert call["config"]["http_options"] == {"timeout": 5000}  # the SDK counts in milliseconds
    assert (result.text, result.model, result.attempts) == ("Use your voice [1].", "m-1", 1)
    assert (result.prompt_tokens, result.completion_tokens, result.thought_tokens, result.total_tokens) == (120, 30, 7, 157)
    assert result.latency_ms >= 0


def test_token_usage_is_logged_without_the_prompt_or_the_answer(caplog):
    client = FakeClient(response("SECRET ANSWER TEXT"))
    with caplog.at_level(logging.INFO, logger="app.services.llm"):
        make(client).generate(system="sys", prompt="SECRET QUESTION TEXT")
    line = next(r.getMessage() for r in caplog.records if "llm: ok" in r.getMessage())
    for expected in ("model=m-1", "prompt_tokens=120", "completion_tokens=30", "thought_tokens=7", "total_tokens=157", "attempts=1"):
        assert expected in line
    assert "SECRET" not in caplog.text  # questions and transcript text must never reach the logs


def test_missing_usage_metadata_is_tolerated():
    client = FakeClient(SimpleNamespace(text="ok", usage_metadata=None, candidates=[]))
    r = make(client).generate(system="s", prompt="p")
    assert r.text == "ok" and r.prompt_tokens is None and r.total_tokens is None


def test_generate_retries_429_then_succeeds_and_reports_attempts():
    sleeps = []
    client = FakeClient(ApiError(429), ApiError(503), response("done"))
    r = make(client, sleeps=sleeps).generate(system="s", prompt="p")
    assert r.text == "done" and r.attempts == 3 and len(client.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_persistent_overload_becomes_llm_unavailable_without_leaking_details():
    client = FakeClient(ApiError(503, "key=SECRET quota"), ApiError(503), ApiError(503, "key=SECRET quota"))
    with pytest.raises(LLMUnavailable) as ei:
        make(client).generate(system="s", prompt="p")
    assert len(client.calls) == 3
    assert "SECRET" not in ei.value.user_message and "503" not in ei.value.user_message


def test_timeouts_are_retried_too():
    client = FakeClient(TimeoutError(), response("ok"))
    assert make(client).generate(system="s", prompt="p").attempts == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried_and_become_llm_rejected(code):
    client = FakeClient(ApiError(code, "API key not valid"))
    with pytest.raises(LLMRejected) as ei:
        make(client).generate(system="s", prompt="p")
    assert len(client.calls) == 1
    assert "API key not valid" not in ei.value.user_message
    assert "LLM_MODEL" in ei.value.user_message  # tells the admin what to check


@pytest.mark.parametrize("bad", [SimpleNamespace(text=None, usage_metadata=None, candidates=[]),
                                 SimpleNamespace(text="   ", usage_metadata=None, candidates=[SimpleNamespace(finish_reason="SAFETY")])])
def test_empty_or_blocked_responses_raise_llm_empty_response(bad):
    with pytest.raises(LLMEmptyResponse):
        make(FakeClient(bad)).generate(system="s", prompt="p")


def test_a_response_whose_text_property_raises_is_treated_as_empty():
    class Broken:
        usage_metadata = None
        candidates = []

        @property
        def text(self):
            raise ValueError("no parts")

    with pytest.raises(LLMEmptyResponse):
        make(FakeClient(Broken())).generate(system="s", prompt="p")


def test_missing_api_key_raises_not_configured_before_any_sdk_import():
    with pytest.raises(LLMNotConfigured) as ei:
        GeminiLLM(model="m", api_key="", timeout_sec=5, max_retries=1, max_output_tokens=10).generate(system="s", prompt="p")
    assert "GEMINI_API_KEY" in ei.value.user_message


def test_every_error_type_has_a_short_user_message():
    for cls in (LLMError, LLMNotConfigured, LLMUnavailable, LLMRejected, LLMEmptyResponse):
        assert 10 < len(cls("internal detail").user_message) < 140
        assert "internal detail" not in cls("internal detail").user_message


# ---- swapping the LLM
def test_set_llm_replaces_and_restores_the_default():
    fake = FakeLLM()
    llm.set_llm(fake)
    try:
        assert llm.get_llm() is fake
    finally:
        llm.set_llm(None)
    assert isinstance(llm.get_llm(), GeminiLLM)


def test_default_llm_is_built_from_settings_without_calling_the_sdk():
    llm._default.cache_clear()
    d = llm._default()
    assert (d.model, d._timeout_sec, d._max_retries, d._max_output_tokens) == ("test-model", 5.0, 2, 256)
    llm._default.cache_clear()
