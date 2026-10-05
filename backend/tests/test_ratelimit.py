import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import redis
from fastapi import HTTPException

from app.api.deps import CurrentUser
from app.core import ratelimit
from app.services import retrieval


class FakeRedis:
    """Just enough of redis-py's pipeline API; no network, so these unit tests never touch real limiter state."""

    def __init__(self):
        self.counts = {}

    def pipeline(self):
        outer = self

        class Pipe:
            def __init__(self):
                self.ops = []

            def incr(self, key):
                self.ops.append(key)

            def expire(self, key, seconds):
                pass

            def execute(self):
                out = []
                for key in self.ops:
                    outer.counts[key] = outer.counts.get(key, 0) + 1
                    out.append(outer.counts[key])
                return out

        return Pipe()


# ---- fixed-window logic (unit)
def test_allows_up_to_limit_then_blocks_with_retry_after():
    r = FakeRedis()
    results = [ratelimit.hit("k", 3, 60, client=r, now=1000.0) for _ in range(5)]
    assert results[:3] == [None] * 3
    assert results[3] == results[4] == 20  # window 960..1020, now=1000 -> 20 s left


def test_window_rollover_resets_the_counter():
    r = FakeRedis()
    for _ in range(4):
        ratelimit.hit("k", 3, 60, client=r, now=1000.0)
    assert ratelimit.hit("k", 3, 60, client=r, now=1000.0) is not None
    assert ratelimit.hit("k", 3, 60, client=r, now=1020.0) is None  # new window


def test_keys_are_independent():
    r = FakeRedis()
    for _ in range(3):
        ratelimit.hit("a", 3, 60, client=r, now=5.0)
    assert ratelimit.hit("a", 3, 60, client=r, now=5.0) is not None
    assert ratelimit.hit("b", 3, 60, client=r, now=5.0) is None


def test_retry_after_is_at_least_one_second():
    r = FakeRedis()
    for _ in range(2):
        ratelimit.hit("k", 1, 60, client=r, now=119.9999)
    assert ratelimit.hit("k", 1, 60, client=r, now=119.9999) == 1


def test_fails_open_when_redis_is_down():
    class Broken:
        def pipeline(self):
            raise redis.ConnectionError("down")

    assert ratelimit.hit("k", 1, 60, client=Broken()) is None


def test_enforce_raises_429_with_retry_after_header():
    with patch.object(ratelimit, "hit", return_value=17):
        with pytest.raises(HTTPException) as ei:
            ratelimit.enforce("x", "u", 5)
    assert ei.value.status_code == 429 and ei.value.headers["Retry-After"] == "17"


def test_ask_limit_is_10_per_minute_per_user():
    r = FakeRedis()
    alice, bob = CurrentUser(id=str(uuid.uuid4()), role="user"), CurrentUser(id=str(uuid.uuid4()), role="user")
    with patch.object(ratelimit, "_redis", return_value=r):
        for _ in range(10):
            ratelimit.ask_rate_limit(alice)
        with pytest.raises(HTTPException) as ei:
            ratelimit.ask_rate_limit(alice)
        ratelimit.ask_rate_limit(bob)  # other users are unaffected
    assert ei.value.status_code == 429 and 1 <= int(ei.value.headers["Retry-After"]) <= 60


# ---- through the real app (integration; Redis db 15 is flushed around every test)
def register_and_login(api, email="ann@example.com"):
    api.post("/auth/register", json={"email": email, "password": "Passw0rd123"})
    r = api.post("/auth/login", json={"email": email, "password": "Passw0rd123"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def mid_window():
    """The limiter is a fixed window: a test whose requests straddle a minute boundary would get a fresh counter and
    flake. Pin the clock to the middle of a window."""
    with patch.object(ratelimit, "time", SimpleNamespace(time=lambda: 1_000_000_020.0)):
        yield


def test_login_limited_to_5_per_minute_per_email(api, mid_window):
    api.post("/auth/register", json={"email": "ann@example.com", "password": "Passw0rd123"})
    bad = {"email": "ann@example.com", "password": "Wrong1234"}
    assert [api.post("/auth/login", json=bad).status_code for _ in range(5)] == [401] * 5
    blocked = api.post("/auth/login", json=bad)
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60
    # a correct password does not bypass the limit ...
    assert api.post("/auth/login", json={"email": "ann@example.com", "password": "Passw0rd123"}).status_code == 429
    # ... and other emails from the same IP have their own bucket
    api.post("/auth/register", json={"email": "bob@example.com", "password": "Passw0rd123"})
    assert api.post("/auth/login", json={"email": "bob@example.com", "password": "Passw0rd123"}).status_code == 200


def test_login_limit_ignores_email_case(api, mid_window):
    bad = {"password": "Wrong1234"}
    for e in ["ann@example.com", "ANN@example.com", "Ann@Example.com", "ann@EXAMPLE.com", "aNN@example.com"]:
        assert api.post("/auth/login", json={"email": e, **bad}).status_code == 401
    assert api.post("/auth/login", json={"email": "ann@example.com", **bad}).status_code == 429


def test_limiter_state_is_flushed_between_tests(api):
    # The previous tests exhausted this bucket; a clean Redis db 15 means it starts fresh.
    bad = {"email": "ann@example.com", "password": "Wrong1234"}
    assert api.post("/auth/login", json=bad).status_code == 401


def test_search_limited_to_30_per_minute_per_user(api, mid_window):
    alice = register_and_login(api, "alice@example.com")
    bob = register_and_login(api, "bob@example.com")
    with patch.object(retrieval, "vector_search", return_value=[]):  # the embedding model is never loaded
        codes = [api.get("/api/search?q=hello", headers=alice).status_code for _ in range(30)]
        assert codes == [200] * 30
        blocked = api.get("/api/search?q=hello", headers=alice)
        assert blocked.status_code == 429 and 1 <= int(blocked.headers["Retry-After"]) <= 60
        assert api.get("/api/search?q=hello", headers=bob).status_code == 200  # per user


def test_search_requires_login_before_rate_limiting(api):
    assert api.get("/api/search?q=hello").status_code == 401
