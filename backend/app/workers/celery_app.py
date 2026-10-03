from celery import Celery

from app.core.config import settings

celery = Celery("svs", broker=settings.REDIS_URL, backend=settings.REDIS_URL, include=["app.workers.tasks"])
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1)
celery.conf.beat_schedule = {
    "purge-expired-refresh-tokens": {"task": "app.workers.tasks.purge_expired_tokens", "schedule": 3600.0},
}
