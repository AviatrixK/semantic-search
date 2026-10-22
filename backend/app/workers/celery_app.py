import logging

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging
from celery.signals import worker_process_init

from app.core.config import settings

celery = Celery("svs", broker=settings.REDIS_URL, backend=settings.REDIS_URL, include=["app.workers.tasks"])
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1)
celery.conf.beat_schedule = {
    "purge-expired-refresh-tokens": {"task": "app.workers.tasks.purge_expired_tokens", "schedule": 3600.0},
}


@celery_setup_logging.connect
def _configure_logging(**_) -> None:
    """Connecting to this signal makes Celery skip its own logging setup (worker AND beat), so both log through
    app.core.logging with the same format and LOG_LEVEL as the API. `--loglevel` on the command line is ignored."""
    from app.core.logging import setup_logging
    setup_logging()


@worker_process_init.connect
def _limit_threads(**_) -> None:
    """Runs in every worker process before its first job: keeps the heavy models from taking every CPU."""
    from app.services.embedding import limit_torch_threads
    try:
        limit_torch_threads(settings.TORCH_THREADS)
    except ImportError:  # an image without the local models (hosted transcription + embeddings only) has no torch to limit
        logging.getLogger(__name__).info("torch is not installed: nothing to limit")
