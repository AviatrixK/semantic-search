"""Per-user daily LLM token budget, kept in Redis (key per user and UTC day, so it resets at midnight UTC by itself).
The check happens before a request starts and between agent steps; one request can overshoot a little (a soft cap).
Fails open: if Redis is down we log and let requests through, like the rate limiter."""
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable

import redis

from app.core.config import settings

log = logging.getLogger(__name__)
DAY_SECONDS = 86400


class DailyBudgetExceeded(Exception):
    """The user used up today's token budget."""

    def __init__(self, retry_after: int):
        super().__init__("daily token budget exhausted")
        self.retry_after = retry_after
        self.user_message = "You have reached today's AI usage limit. It resets at midnight UTC, so please try again then."


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


def estimate_tokens(text: str) -> int:
    """Rough size when the API did not report usage: about 4 characters per token."""
    return max(1, len(text) // 4)


class UsageMeter:
    def __init__(self, *, client=None, clock: Callable[[], float] = time.time, budget: int | None = None):
        self._client = client
        self._clock = clock
        self._budget = settings.DAILY_TOKEN_BUDGET if budget is None else budget

    def _redis(self):
        return self._client or _redis()

    def key(self, user_id: str) -> str:
        day = datetime.fromtimestamp(self._clock(), tz=timezone.utc).strftime("%Y%m%d")
        return f"tokens:{user_id}:{day}"

    def seconds_until_reset(self) -> int:
        now = self._clock()
        return max(1, int(DAY_SECONDS - now % DAY_SECONDS))

    def used(self, user_id: str) -> int:
        try:
            return int(self._redis().get(self.key(user_id)) or 0)
        except redis.RedisError:
            log.warning("usage meter: redis unavailable (read), allowing")
            return 0

    def exceeded(self, user_id: str) -> bool:
        return self._budget > 0 and self.used(user_id) >= self._budget

    def check(self, user_id: str) -> None:
        if self.exceeded(user_id):
            raise DailyBudgetExceeded(self.seconds_until_reset())

    def add(self, user_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        try:
            pipe = self._redis().pipeline()
            key = self.key(user_id)
            pipe.incrby(key, tokens)
            pipe.expire(key, 2 * DAY_SECONDS)  # outlives its day, then disappears
            pipe.execute()
        except redis.RedisError:
            log.warning("usage meter: redis unavailable (write), tokens not counted")
