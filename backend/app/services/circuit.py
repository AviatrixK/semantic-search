"""A small circuit breaker for LLM models, with its state in Redis so every gunicorn worker (and the Celery worker) shares it.

After `threshold` CONSECUTIVE 503s from a model the circuit opens for `cooldown` seconds: callers skip that model and go
straight to the next one instead of waiting through retries that are very likely to fail again.

  llm:cb:fail:<model>   number of 503s since the last success (expires after 5 minutes of quiet)
  llm:cb:open:<model>   exists while the circuit is open (Redis expires it after the cooldown)

When the cooldown ends the model is tried again. The failure count is deliberately NOT reset when the circuit opens, so if
that probe fails with a 503 the circuit re-opens at once (half-open behaviour); a success resets everything.

Fail-open: if Redis is unreachable the breaker never blocks anything (it just behaves as if it did not exist)."""
import logging
from functools import lru_cache

import redis

from app.core.config import settings

log = logging.getLogger(__name__)
FAIL_TTL_SEC = 300


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


class CircuitBreaker:
    def __init__(self, *, client=None, threshold: int | None = None, cooldown_sec: int | None = None, prefix: str = "llm:cb"):
        self._client = client
        self.threshold = settings.LLM_BREAKER_THRESHOLD if threshold is None else threshold
        self.cooldown_sec = settings.LLM_BREAKER_COOLDOWN_SEC if cooldown_sec is None else cooldown_sec
        self._prefix = prefix

    def _r(self):
        return self._client or _redis()

    def _fail_key(self, name: str) -> str:
        return f"{self._prefix}:fail:{name}"

    def _open_key(self, name: str) -> str:
        return f"{self._prefix}:open:{name}"

    def is_open(self, name: str) -> bool:
        try:
            return self._r().get(self._open_key(name)) is not None
        except redis.RedisError:
            log.warning("llm circuit breaker: redis unavailable, treating %s as closed", name)
            return False

    def record_failure(self, name: str) -> bool:
        """Count one 503. Returns True if this opened (or re-opened) the circuit."""
        try:
            r = self._r()
            count = int(r.incrby(self._fail_key(name), 1))
            r.expire(self._fail_key(name), FAIL_TTL_SEC)
            if count >= self.threshold:
                r.set(self._open_key(name), "1", ex=self.cooldown_sec)
                log.warning("llm circuit breaker: %s opened for %ds after %d consecutive 503s", name, self.cooldown_sec, count)
                return True
        except redis.RedisError:
            log.warning("llm circuit breaker: redis unavailable, 503 from %s not counted", name)
        return False

    def record_success(self, name: str) -> None:
        try:
            self._r().delete(self._fail_key(name), self._open_key(name))
        except redis.RedisError:
            pass
