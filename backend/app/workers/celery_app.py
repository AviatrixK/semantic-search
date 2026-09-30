from celery import Celery

from app.core.config import settings

celery = Celery("svs", broker=settings.REDIS_URL, backend=settings.REDIS_URL, include=["app.workers.tasks"])
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1)
