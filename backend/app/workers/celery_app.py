from celery import Celery
from celery.signals import worker_process_init

from app.core.config import settings

celery = Celery("svs", broker=settings.REDIS_URL, backend=settings.REDIS_URL, include=["app.workers.tasks"])
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1)
celery.conf.beat_schedule = {
    "purge-expired-refresh-tokens": {"task": "app.workers.tasks.purge_expired_tokens", "schedule": 3600.0},
}


@worker_process_init.connect
def _limit_threads(**_) -> None:
    """Runs in every worker process before its first job: keeps the heavy models from taking every CPU."""
    from app.services.embedding import limit_torch_threads
    limit_torch_threads(settings.TORCH_THREADS)
