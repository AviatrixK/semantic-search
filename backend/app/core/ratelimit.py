"""Small Redis fixed-window rate limiter (no extra library), exposed as FastAPI dependencies.

Each (name, identity) pair gets one counter per window: the key embeds the window number, so counters
expire by themselves. If Redis is unreachable the limiter fails open (logs a warning) rather than taking
login and search down with it.
"""
import logging
import math
import time
from functools import lru_cache

import redis
from fastapi import Depends, HTTPException, Request

from app.api.deps import CurrentUser, current_user
from app.core.config import settings
from app.schemas.auth import LoginIn

log = logging.getLogger(__name__)
WINDOW_SEC = 60


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


def hit(key: str, limit: int, window: int = WINDOW_SEC, *, client=None, now: float | None = None) -> int | None:
    """Count one request. Returns None if allowed, else the seconds until the window resets (>= 1)."""
    now = time.time() if now is None else now
    bucket = int(now // window)
    redis_key = f"rl:{key}:{bucket}"
    try:
        pipe = (client or _redis()).pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, window)
        count = pipe.execute()[0]
    except redis.RedisError:
        log.warning("rate limiter: redis unavailable, allowing %s", key)
        return None
    if count > limit:
        return max(1, math.ceil((bucket + 1) * window - now))
    return None


def enforce(name: str, identity: str, limit: int) -> None:
    retry_after = hit(f"{name}:{identity}", limit)
    if retry_after is not None:
        raise HTTPException(429, f"Too many requests. Try again in {retry_after} seconds.",
                            headers={"Retry-After": str(retry_after)})


def login_rate_limit(request: Request, body: LoginIn) -> None:
    """Per client IP + email. Counts every attempt, so a correct password does not bypass the limit.
    (Behind a reverse proxy, configure it to pass the real client address to uvicorn.)"""
    ip = request.client.host if request.client else "unknown"
    enforce("login", f"{ip}:{body.email.lower()}", settings.RATE_LOGIN_PER_MIN)


def search_rate_limit(user: CurrentUser = Depends(current_user)) -> None:
    enforce("search", user.id, settings.RATE_SEARCH_PER_MIN)


def ask_rate_limit(user: CurrentUser = Depends(current_user)) -> None:
    """For /api/ask (Phase 8)."""
    enforce("ask", user.id, settings.RATE_ASK_PER_MIN)
