"""Job progress as a live feed: the worker and the upload route PUBLISH every change of a job to a Redis pub/sub channel,
and GET /api/jobs/events (admin only) relays it as Server-Sent Events. No client ever polls.

Pub/sub keeps no history, so a new subscriber first receives a snapshot of every active job (see `active_jobs`)."""
import json
import logging
from functools import lru_cache

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Job

log = logging.getLogger(__name__)
CHANNEL = "svs:job-events"
ACTIVE_STAGES = ("queued", "extracting", "transcribing", "embedding")


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


def payload(job_id, video_id, stage: str, progress: int, error: str | None = None) -> dict:
    """The one shape used by the snapshot, the live events and GET /api/jobs/{id}."""
    return {"id": str(job_id), "video_id": str(video_id), "stage": stage, "progress": progress, "error": error}


def publish(event: dict, *, client=None) -> None:
    """Fire and forget. A Redis problem must never fail a job or an upload: the job state is in Postgres, the feed
    is only a convenience (the next page load or reconnect sends the snapshot again)."""
    try:
        (client or _redis()).publish(CHANNEL, json.dumps(event, separators=(",", ":")))
    except redis.RedisError:
        log.warning("job event not published: redis unavailable (job %s)", event.get("id"))


def active_jobs(db: Session) -> list[dict]:
    """Current state of every job that is still running, oldest first."""
    rows = db.scalars(select(Job).where(Job.stage.in_(ACTIVE_STAGES)).order_by(Job.updated_at)).all()
    return [payload(j.id, j.video_id, j.stage, j.progress, j.error) for j in rows]
