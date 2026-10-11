"""GET /api/jobs/events: live job progress for the admin page (Server-Sent Events).

Registered BEFORE the videos router: that one has /api/jobs/{job_id}, which would otherwise try to read "events" as a UUID."""
import json
import logging
from collections.abc import AsyncIterator, Callable

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError

from app.api.deps import CurrentUser, require_admin
from app.core.config import settings
from app.core.db import SessionLocal
from app.services import jobevents

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["jobs"])


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def load_snapshot() -> list[dict]:
    with SessionLocal() as db:  # short session: it must not stay open for as long as the stream does
        return jobevents.active_jobs(db)


async def _close_quietly(*closers) -> None:
    for close in closers:
        try:
            await close()
        except Exception:  # cleanup must not hide the real reason the stream ended
            log.debug("closing job event stream", exc_info=True)


async def job_stream(client, pubsub, snapshot: list[dict], keepalive_sec: float) -> AsyncIterator[str]:
    """snapshot, then one `job` event per change, with a comment line whenever the channel is idle. Cleans up when the
    client goes away (Starlette cancels this generator) or Redis fails (the stream ends; the browser reconnects)."""
    try:
        yield sse("snapshot", {"jobs": snapshot})
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=keepalive_sec)
            if message is None:
                yield ": keep-alive\n\n"  # ignored by clients; keeps proxies from closing an idle stream
                continue
            data = message["data"]
            yield f"event: job\ndata: {data.decode() if isinstance(data, bytes) else data}\n\n"
    except RedisError:
        log.warning("job event stream ended: redis error")
    finally:
        await _close_quietly(pubsub.aclose, client.aclose)


def get_redis_factory() -> Callable[[], aioredis.Redis]:
    return lambda: aioredis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)


@router.get("/jobs/events")
async def job_events(_: CurrentUser = Depends(require_admin), make_client: Callable = Depends(get_redis_factory)):
    client = make_client()
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(jobevents.CHANNEL)  # subscribe BEFORE the snapshot, so no change falls between the two
        snapshot = await run_in_threadpool(load_snapshot)
    except Exception as exc:
        await _close_quietly(pubsub.aclose, client.aclose)
        log.warning("job events unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "Live updates are unavailable right now.")  # a real status: the client retries with backoff
    return StreamingResponse(
        job_stream(client, pubsub, snapshot, settings.SSE_KEEPALIVE_SEC), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},  # no buffering by any proxy
    )
