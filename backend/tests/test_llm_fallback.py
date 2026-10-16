"""Fallback model chain and circuit breaker, driven through the REAL GeminiLLM with a fake SDK client (no network, no key).

Every path of the spec: primary answers; primary fails (503/504/429/timeout) and a fallback answers once; the circuit opens
after 3 consecutive 503s, is shared through Redis, closes after the cooldown; every model fails and the error says why."""
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.services import llm
from app.services.circuit import CircuitBreaker
from app.services.llm import (FallbackLLM, GeminiLLM, LLMOverloaded, LLMQuotaExceeded, LLMRejected, LLMUnavailable, ModelMsg,
                              ToolCall, UserMsg, build_chain)
from tests.fakes import FakeRedis
from tests.test_llm import ApiError, FakeClient, response

PRIMARY, FB1, FB2 = "test-model", "fb-one", "fb-two"


def turn_response(text="ok"):
    part = SimpleNamespace(text=text, function_call=None, thought=False)
    return SimpleNamespace(text=text, usage_metadata=None, candidates=[SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[part]))])


class Chain:
    """build_chain() with fake clients per model name. Records sleeps (retry backoff) so no test waits."""

    def __init__(self, clients: dict, fallbacks=(FB1,), redis=None, retries=2):
        self.clients, self.redis, self.sleeps = clients, redis or FakeRedis(), []
        self.breaker = CircuitBreaker(client=self.redis, threshold=3, cooldown_sec=60)

        def make(**kw):
            return GeminiLLM(**kw, client=self.clients[kw["model"]], sleep=self.sleeps.append, jitter=lambda: 1.0)

        with patch.object(settings, "LLM_FALLBACK_MODELS", ",".join(fallbacks)), patch.object(settings, "LLM_MAX_RETRIES", retries):
            self.llm = build_chain(breaker=self.breaker, make=make)

    def ask(self):
        return self.llm.generate(system="s", prompt="p")


def overloaded(n):
    return [ApiError(503, "The model is overloaded. Please try again later.") for _ in range(n)]


# ---------------------------------------------------------------- wiring
def test_chain_is_primary_then_fallbacks_with_one_attempt_each():
    c = Chain({PRIMARY: FakeClient(), FB1: FakeClient(), FB2: FakeClient()}, fallbacks=(FB1, FB2))
    assert [n for n, _ in c.llm.models] == [PRIMARY, FB1, FB2]
    assert [m._max_retries for _, m in c.llm.models] == [2, 0, 0]  # the primary keeps LLM_MAX_RETRIES; fallbacks get ONE attempt


def test_fallback_setting_is_cleaned():
    for raw, expected in [("", []), ("  ", []), ("a,b", ["a", "b"]), (" a , ,b,a ", ["a", "b"]), (f"{PRIMARY},a", ["a"])]:
        with patch.object(settings, "LLM_FALLBACK_MODELS", raw):
            assert settings.llm_fallback_models == expected, raw


# ---------------------------------------------------------------- the paths
def test_primary_answers_and_fallbacks_are_never_called():
    primary, fb = FakeClient(response("from primary")), FakeClient()
    r = Chain({PRIMARY: primary, FB1: fb}).ask()
    assert r.text == "from primary" and r.model == PRIMARY and fb.calls == []


def test_primary_503_after_its_retries_then_the_fallback_answers_once(caplog):
    primary, fb = FakeClient(*overloaded(3)), FakeClient(response("from fallback"))
    c = Chain({PRIMARY: primary, FB1: fb})
    with caplog.at_level(logging.INFO):
        r = c.ask()
    assert r.text == "from fallback" and r.model == FB1  # the result says which model answered
    assert len(primary.calls) == 3 and len(fb.calls) == 1  # 1 try + 2 retries, then ONE fallback call
    assert c.sleeps == [1.0, 2.0]
    assert any(f"answered by fallback model {FB1}" in m for m in caplog.messages)  # and so does the log


def test_a_fallback_is_tried_once_not_retried():
    primary, fb1, fb2 = FakeClient(*overloaded(3)), FakeClient(*overloaded(1)), FakeClient(response("third"))
    c = Chain({PRIMARY: primary, FB1: fb1, FB2: fb2}, fallbacks=(FB1, FB2))
    r = c.ask()
    assert r.model == FB2 and len(fb1.calls) == 1 and len(fb2.calls) == 1


@pytest.mark.parametrize("failure,kind", [(ApiError(504), LLMUnavailable), (TimeoutError(), LLMUnavailable),
                                          (ApiError(429), LLMQuotaExceeded), (ApiError(503), LLMOverloaded)])
def test_these_failures_all_fall_through_to_the_fallback(failure, kind):
    primary, fb = FakeClient(*[failure] * 3), FakeClient(response("fb"))
    assert Chain({PRIMARY: primary, FB1: fb}).ask().model == FB1


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_a_rejected_request_is_not_an_outage_and_does_not_fall_through(code):
    primary, fb = FakeClient(ApiError(code, "API key not valid")), FakeClient(response("fb"))
    with pytest.raises(LLMRejected):
        Chain({PRIMARY: primary, FB1: fb}).ask()
    assert fb.calls == []  # hiding a bad key or a retired model behind a fallback would make it invisible


def test_every_model_failing_raises_the_first_failure_with_its_own_message():
    c = Chain({PRIMARY: FakeClient(*overloaded(3)), FB1: FakeClient(ApiError(429))})
    with pytest.raises(LLMOverloaded) as e:
        c.ask()
    assert e.value.status == 503 and "overloaded" in e.value.user_message and "usage limit" not in e.value.user_message


def test_error_classes_tell_overloaded_quota_and_unreachable_apart():
    over = Chain({PRIMARY: FakeClient(*overloaded(3))}, fallbacks=())
    quota = Chain({PRIMARY: FakeClient(*[ApiError(429)] * 3)}, fallbacks=())
    gone = Chain({PRIMARY: FakeClient(*[TimeoutError()] * 3)}, fallbacks=())
    errors = []
    for c in (over, quota, gone):
        with pytest.raises(LLMUnavailable) as e:
            c.ask()
        errors.append(e.value)
    assert [type(e) for e in errors] == [LLMOverloaded, LLMQuotaExceeded, LLMUnavailable]
    assert len({e.user_message for e in errors}) == 3 and len({e.degraded_reason for e in errors}) == 3
    assert all("SECRET" not in e.user_message and 10 < len(e.user_message) < 140 for e in errors)
    assert "overloaded" in errors[0].user_message and "usage limit" in errors[1].user_message


# ---------------------------------------------------------------- circuit breaker
def test_three_consecutive_503s_open_the_circuit_and_the_next_call_skips_the_model():
    primary, fb = FakeClient(*overloaded(3)), FakeClient(response("a"), response("b"))
    c = Chain({PRIMARY: primary, FB1: fb})
    c.ask()  # the primary's three 503 attempts open its circuit
    assert c.breaker.is_open(PRIMARY) and c.redis.ttl[f"llm:cb:open:{PRIMARY}"] == 60  # for 60 seconds
    c.sleeps.clear()
    r = c.ask()  # straight to the fallback: no primary call, no retry sleeps
    assert r.model == FB1 and len(primary.calls) == 3 and c.sleeps == []


def test_two_503s_then_a_success_never_open_it_and_the_count_resets():
    primary = FakeClient(*overloaded(2), response("ok"), *overloaded(2), response("ok again"))
    c = Chain({PRIMARY: primary, FB1: FakeClient()})
    assert c.ask().text == "ok" and c.ask().text == "ok again"  # 2 + success, then 2 + success: never 3 in a row
    assert not c.breaker.is_open(PRIMARY)


def test_other_errors_do_not_count_toward_the_circuit():
    primary = FakeClient(*[ApiError(504)] * 3, *[ApiError(429)] * 3)
    c = Chain({PRIMARY: primary, FB1: FakeClient(response("1"), response("2"))})
    c.ask()
    c.ask()
    assert not c.breaker.is_open(PRIMARY)  # only 503s are counted


def test_after_the_cooldown_the_model_is_probed_and_one_more_503_reopens_it():
    primary = FakeClient(*overloaded(3), ApiError(503), response("recovered"))
    c = Chain({PRIMARY: primary, FB1: FakeClient(*[response("fb")] * 4)}, retries=0)
    c.llm.breaker.threshold = 3
    for _ in range(3):  # three calls, one 503 each (no retries here): the third opens the circuit
        c.ask()
    assert c.breaker.is_open(PRIMARY)
    c.redis.delete(f"llm:cb:open:{PRIMARY}")  # Redis expired the key: cooldown over
    assert c.ask().model == FB1 and len(primary.calls) == 4  # the probe went to the primary, it 503'd...
    assert c.breaker.is_open(PRIMARY)  # ...so the circuit is open again at once (half-open behaviour)
    c.redis.delete(f"llm:cb:open:{PRIMARY}")
    assert c.ask().model == PRIMARY  # the next probe succeeds
    assert not c.breaker.is_open(PRIMARY) and f"llm:cb:fail:{PRIMARY}" not in c.redis.data  # and everything is reset


def test_the_circuit_is_shared_by_every_worker_through_redis():
    shared = FakeRedis()
    worker_a = Chain({PRIMARY: FakeClient(*overloaded(3)), FB1: FakeClient(response("a"))}, redis=shared)
    worker_a.ask()
    other_primary = FakeClient(response("never called"))
    worker_b = Chain({PRIMARY: other_primary, FB1: FakeClient(response("b"))}, redis=shared)  # a second gunicorn worker
    assert worker_b.ask().model == FB1 and other_primary.calls == []


def test_each_model_has_its_own_circuit():
    c = Chain({PRIMARY: FakeClient(*overloaded(3)), FB1: FakeClient(response("fine"))})
    c.ask()
    assert c.breaker.is_open(PRIMARY) and not c.breaker.is_open(FB1)


def test_all_circuits_open_fails_fast_without_calling_anyone():
    primary, fb = FakeClient(), FakeClient()
    c = Chain({PRIMARY: primary, FB1: fb})
    for name in (PRIMARY, FB1):
        c.redis.set(f"llm:cb:open:{name}", "1", ex=60)
    with pytest.raises(LLMOverloaded):
        c.ask()
    assert primary.calls == [] and fb.calls == []


def test_a_dead_redis_never_blocks_a_model():
    c = Chain({PRIMARY: FakeClient(*overloaded(3), response("fine")), FB1: FakeClient(response("fb"))})
    c.redis.down = True
    assert c.ask().model == FB1  # failover still works without the breaker
    assert c.ask().model == PRIMARY  # and the primary was never skipped (fail-open)


# ---------------------------------------------------------------- tool-calling turns
def test_fallback_turns_get_plain_messages_without_the_primarys_provider_state():
    primary, fb = FakeClient(*overloaded(3)), FakeClient(turn_response("answer"))
    c = Chain({PRIMARY: primary, FB1: fb})
    history = [UserMsg("q"), ModelMsg(text=None, tool_calls=(ToolCall("search_transcripts", {"query": "x"}),),
                                      raw={"role": "model", "parts": ["thought signature of the primary"]})]
    turn = c.llm.generate_turn(system="s", messages=history)
    assert turn.text == "answer" and turn.model == FB1
    sent_to_primary = primary.calls[0]["contents"][1]
    sent_to_fallback = fb.calls[0]["contents"][1]
    assert "thought signature of the primary" in str(sent_to_primary)  # the primary got its own object back
    assert "thought signature" not in str(sent_to_fallback) and sent_to_fallback["parts"][0]["function_call"]["name"] == "search_transcripts"


# ---------------------------------------------------------------- misc
def test_the_primary_name_is_exposed_for_compatibility():
    assert Chain({PRIMARY: FakeClient(), FB1: FakeClient()}).llm.model == PRIMARY


def test_an_empty_chain_is_a_programming_error():
    with pytest.raises(ValueError):
        FallbackLLM([], CircuitBreaker(client=FakeRedis()))


def test_default_chain_comes_from_settings_and_the_sdk_is_not_touched():
    llm._default.cache_clear()
    try:
        with patch.object(settings, "LLM_FALLBACK_MODELS", FB1):
            chain = llm._default()
        assert [n for n, _ in chain.models] == [PRIMARY, FB1]
        assert all(m._client is None for _, m in chain.models)  # the genai client is created lazily, on first use
    finally:
        llm._default.cache_clear()
