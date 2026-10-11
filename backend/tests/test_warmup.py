"""API startup: the embedding model is loaded and used once BEFORE the app reports ready, and the worker's thread limit
never reaches the API."""
import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import embedding, storage


@pytest.fixture(autouse=True)
def _reset_state():
    main.app.state.ready = True
    embedding._warm = False
    yield
    main.app.state.ready = True
    embedding._warm = False


def run_lifespan(events: list):
    async def go():
        async with main.lifespan(main.app):
            events.append("serving")
    asyncio.run(go())


def test_warm_up_loads_the_model_and_encodes_once():
    model = MagicMock()
    model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
    with patch.object(embedding, "_model", return_value=model) as load:
        ms = embedding.warm_up()
    assert load.called and model.encode.call_count == 1
    assert embedding.is_warm() and ms >= 0


def test_model_is_loaded_before_the_app_reports_ready():
    events = []
    seen_ready = []

    def fake_warm_up():
        seen_ready.append(main.app.state.ready)  # what /health would see right now
        events.append("model loaded + warmed")
        return 1.0

    with patch.object(storage, "ensure_bucket", side_effect=lambda: events.append("bucket")), \
         patch.object(embedding, "warm_up", side_effect=fake_warm_up):
        run_lifespan(events)
    assert events == ["bucket", "model loaded + warmed", "serving"]
    assert seen_ready == [False]  # not ready while warming
    assert main.app.state.ready is True  # ready once it finished


def test_health_is_503_while_warming_and_200_after():
    client = TestClient(main.app)
    main.app.state.ready = False
    r = client.get("/health")
    assert r.status_code == 503 and "model" in r.json()["detail"]
    main.app.state.ready = True
    assert client.get("/health").json() == {"ok": True}


def test_a_failed_warm_up_never_reports_ready():
    with patch.object(storage, "ensure_bucket"), \
         patch.object(embedding, "warm_up", side_effect=RuntimeError("model download failed")):
        with pytest.raises(RuntimeError, match="download"):
            run_lifespan([])
    assert main.app.state.ready is False  # the worker dies; the healthcheck keeps failing


def test_warm_up_can_be_switched_off_for_development():
    with patch.object(storage, "ensure_bucket"), patch.object(embedding, "warm_up") as warm, \
         patch.object(main.settings, "WARMUP_ON_START", False):
        run_lifespan([])
    warm.assert_not_called()
    assert main.app.state.ready is True


def test_api_startup_never_applies_the_worker_thread_limit():
    with patch.object(storage, "ensure_bucket"), patch.object(embedding, "limit_torch_threads") as limit, \
         patch.object(embedding, "_model", return_value=MagicMock(encode=lambda *a, **k: np.zeros((1, 384)))):
        run_lifespan([])
    limit.assert_not_called()


def test_only_the_celery_worker_calls_limit_torch_threads():
    """TORCH_THREADS / WHISPER_THREADS are in the shared .env.prod, so the api container sees them too: they must be
    read only by worker code paths."""
    app_dir = Path(main.__file__).parent
    callers = {str(p.relative_to(app_dir)).replace("\\", "/") for p in app_dir.rglob("*.py")
               if re.search(r"limit_torch_threads\(", p.read_text(encoding="utf-8")) and p.name != "embedding.py"}
    assert callers == {"workers/celery_app.py"}
    users = {str(p.relative_to(app_dir)).replace("\\", "/") for p in app_dir.rglob("*.py")
             if re.search(r"WHISPER_THREADS|TORCH_THREADS", p.read_text(encoding="utf-8"))}
    assert users == {"core/config.py", "services/transcription.py", "workers/celery_app.py"}
