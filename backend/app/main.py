import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, jobs, search, videos
from app.core.config import settings
from app.core.logging import setup_logging
from app.services import embedders, embedding, storage

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()  # first: gunicorn leaves the root logger at WARNING, which would hide every log.info below
    app.state.ready = False  # /health answers 503 until everything below is done
    storage.ensure_bucket()
    embedder = embedders.get_embedder()
    embedder.check_configured()  # a hosted provider without its API key stops the worker now, not at the first search
    if embedder.uses_legacy_column and settings.WARMUP_ON_START:
        # (a hosted provider has no model to load: nothing to warm.) Runs in EVERY gunicorn worker (each is its own process with its own model). uvicorn accepts no request until this
        # startup finishes, so the first search never pays the model load. A failure here stops the worker loudly.
        ms = await run_in_threadpool(embedding.warm_up)
        log.info("embedding model warm (%.0f ms, torch threads=%s)", ms, embedding.warm_threads())
    app.state.ready = True
    yield


app = FastAPI(title="Semantic Video Search", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Search-Ms", "Server-Timing", "X-Search-Highlight", "X-Search-Fallback", "Retry-After"])  # let the browser read these
app.include_router(auth.router)
app.include_router(jobs.router)  # before videos: /api/jobs/events must win over /api/jobs/{job_id}
app.include_router(videos.router)
app.include_router(search.router)


@app.get("/health")
def health():
    # `ready` is False only while the lifespan above is running (tests that skip the lifespan never set it).
    if not getattr(app.state, "ready", True):
        raise HTTPException(503, "Starting up: the embedding model is loading")
    return {"ok": True}
