"""CircuitBreaker on its own (the chain tests cover it end to end)."""
from app.services.circuit import FAIL_TTL_SEC, CircuitBreaker
from tests.fakes import FakeRedis


def make(**kw):
    r = FakeRedis()
    return CircuitBreaker(client=r, threshold=3, cooldown_sec=60, **kw), r


def test_opens_exactly_at_the_threshold_for_the_cooldown():
    cb, r = make()
    assert [cb.record_failure("m") for _ in range(3)] == [False, False, True]
    assert cb.is_open("m") and r.ttl["llm:cb:open:m"] == 60  # Redis expires it: nothing to clean up, any worker sees it


def test_failure_counter_expires_when_503s_are_not_consecutive_in_time():
    cb, r = make()
    cb.record_failure("m")
    assert r.ttl["llm:cb:fail:m"] == FAIL_TTL_SEC


def test_a_success_resets_the_count_and_closes_the_circuit():
    cb, r = make()
    for _ in range(3):
        cb.record_failure("m")
    cb.record_success("m")
    assert not cb.is_open("m") and "llm:cb:fail:m" not in r.data
    assert [cb.record_failure("m") for _ in range(2)] == [False, False]  # counting starts again from zero


def test_after_opening_one_more_failure_reopens_without_counting_to_three_again():
    cb, r = make()
    for _ in range(3):
        cb.record_failure("m")
    r.delete("llm:cb:open:m")  # cooldown over
    assert not cb.is_open("m") and cb.record_failure("m") is True and cb.is_open("m")


def test_models_are_independent_and_settings_supply_the_defaults():
    cb, r = make()
    for _ in range(3):
        cb.record_failure("a")
    assert cb.is_open("a") and not cb.is_open("b")
    default = CircuitBreaker(client=r)
    assert (default.threshold, default.cooldown_sec) == (3, 60)  # LLM_BREAKER_THRESHOLD / LLM_BREAKER_COOLDOWN_SEC


def test_redis_down_is_fail_open_and_never_raises():
    cb, r = make()
    r.down = True
    assert cb.is_open("m") is False
    assert cb.record_failure("m") is False
    cb.record_success("m")
