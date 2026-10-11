from unittest.mock import MagicMock, patch

from app.services import embedding
from app.workers import celery_app


def test_limit_torch_threads_caps_torch():
    torch = MagicMock()
    embedding.limit_torch_threads(1, torch_module=torch)
    torch.set_num_threads.assert_called_once_with(1)
    embedding.limit_torch_threads(0, torch_module=torch)
    torch.set_num_threads.assert_called_with(1)  # never 0


def test_worker_process_start_applies_the_configured_limit():
    with patch.object(embedding, "limit_torch_threads") as limit, patch.object(celery_app.settings, "TORCH_THREADS", 3):
        celery_app._limit_threads()
    limit.assert_called_once_with(3)


def test_worker_process_init_signal_is_wired_to_it():
    from celery.signals import worker_process_init
    with patch.object(embedding, "limit_torch_threads") as limit:
        worker_process_init.send(sender=None)
    limit.assert_called_with(celery_app.settings.TORCH_THREADS)
