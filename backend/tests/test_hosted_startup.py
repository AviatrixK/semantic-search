"""Startup with a hosted embedding provider: nothing to warm up, a missing key stops the worker loudly, and the app (API and worker
code) runs with the local ML packages (torch, sentence-transformers, faster-whisper) completely absent, which is what makes it possible
to drop them from the images once the hosted defaults are chosen."""
import asyncio
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app import main
from app.core.config import settings
from app.services import embedders, embedding, storage
from app.services.embedders import EmbeddingNotConfigured, GeminiEmbedder
from app.workers import celery_app

BACKEND = Path(main.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _state():
    main.app.state.ready = True
    yield
    main.app.state.ready = True


def start():
    async def go():
        async with main.lifespan(main.app):
            pass
    asyncio.run(go())


def test_a_hosted_provider_has_nothing_to_warm_up_and_the_app_is_ready():
    emb = GeminiEmbedder(model="m", dim=8, api_key="k")
    with patch.object(storage, "ensure_bucket"), patch.object(embedders, "get_embedder", return_value=emb), \
         patch.object(embedding, "warm_up") as warm:
        start()
    warm.assert_not_called()
    assert main.app.state.ready is True


def test_a_hosted_provider_without_its_key_stops_startup_and_never_reports_ready():
    emb = GeminiEmbedder(model="m", dim=8, api_key="")
    with patch.object(storage, "ensure_bucket"), patch.object(embedders, "get_embedder", return_value=emb), \
         pytest.raises(EmbeddingNotConfigured, match="GEMINI_API_KEY"):
        start()
    assert main.app.state.ready is False  # /health keeps answering 503: the container shows as unhealthy


def test_the_local_provider_still_warms_up_as_before():
    with patch.object(storage, "ensure_bucket"), patch.object(embedding, "warm_up", return_value=1.0) as warm:
        start()
    warm.assert_called_once()


def test_a_worker_without_torch_starts_instead_of_crashing(caplog):
    with patch.object(embedding, "limit_torch_threads", side_effect=ImportError("No module named 'torch'")), \
         caplog.at_level(logging.INFO):
        celery_app._limit_threads()  # must not raise
    assert any("torch is not installed" in m for m in caplog.messages)


def test_the_hosted_code_path_imports_and_starts_with_no_local_ml_packages_installed():
    script = textwrap.dedent("""
        import sys, importlib.abc
        BLOCKED = ("torch", "sentence_transformers", "faster_whisper", "transformers", "ctranslate2", "onnxruntime")
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"{name} is blocked: this process must not need it")
        sys.meta_path.insert(0, Block())

        import asyncio
        from unittest.mock import patch
        from app import main
        import app.workers.tasks, app.workers.celery_app, app.services.retrieval, app.services.sarvam, app.services.vectorstore  # noqa
        from app.services import embedders, storage

        async def go():
            with patch.object(storage, "ensure_bucket"):
                async with main.lifespan(main.app):
                    print("ready", main.app.state.ready, embedders.get_embedder().key)
        asyncio.run(go())
        app.workers.celery_app._limit_threads()
        loaded = [m for m in sys.modules if m.split(".")[0] in BLOCKED]
        print("local ML modules loaded:", loaded)
    """)
    env = dict(os.environ, EMBED_PROVIDER="gemini", TRANSCRIBER="sarvam", GEMINI_API_KEY="k", SARVAM_API_KEY="k", PYTHONPATH=str(BACKEND))
    done = subprocess.run([sys.executable, "-c", script], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=90)
    assert done.returncode == 0, done.stderr[-1500:]
    assert f"ready True gemini/{settings.GEMINI_EMBED_MODEL}@{settings.EMBED_DIMENSIONS}" in done.stdout
    assert "local ML modules loaded: []" in done.stdout


def test_importing_the_local_stack_is_really_what_the_blocker_catches():
    """Guards the test above: with the default (local) provider the same blocked start does need torch, so the block is real."""
    script = textwrap.dedent("""
        import sys, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("torch", "sentence_transformers"):
                    raise ImportError("blocked")
        sys.meta_path.insert(0, Block())
        from app.services import embedding
        embedding.embed_batch(["x"])
    """)
    env = dict(os.environ, EMBED_PROVIDER="local", PYTHONPATH=str(BACKEND))
    done = subprocess.run([sys.executable, "-c", script], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=90)
    assert done.returncode != 0 and "blocked" in done.stderr
