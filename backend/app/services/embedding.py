import hashlib
import logging
import time
from functools import lru_cache

import numpy as np
import redis

from app.core.config import settings

log = logging.getLogger(__name__)


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(settings.EMBED_MODEL)
    dim = m.get_sentence_embedding_dimension()
    if dim != settings.EMBED_DIM:
        raise RuntimeError(f"EMBED_DIM={settings.EMBED_DIM} but model outputs {dim}; fix .env and schema")
    return m


def limit_torch_threads(n: int, torch_module=None) -> None:
    """Cap torch's CPU threads (the worker calls this so one CPU stays free for the API while a video is processed)."""
    torch = torch_module
    if torch is None:
        import torch
    torch.set_num_threads(max(1, n))


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


_warm = False


def warm_up() -> float:
    """Loads the embedding model and runs one encode, so the first real query pays neither the model load (seconds) nor the
    first-call initialisation. Returns the milliseconds it took. Called by the API at startup, in every gunicorn worker.
    Deliberately does NOT touch thread limits: those belong to the Celery worker (see limit_torch_threads)."""
    global _warm
    started = time.perf_counter()
    embed_batch(["warm up"])  # _model() loads on first use; encoding initialises the tokenizer and thread pools
    _warm = True
    return (time.perf_counter() - started) * 1000


def is_warm() -> bool:
    return _warm


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, batch_size=64, normalize_embeddings=True).tolist()


def cache_key(query: str) -> str:
    """sha256(model + query). Whitespace is normalised so "a  b " and "a b" share an entry."""
    normalised = " ".join(query.split())
    return "emb:" + hashlib.sha256(f"{settings.EMBED_MODEL}\n{normalised}".encode()).hexdigest()


def embed_query(text: str, *, client=None) -> list[float]:
    """Query embedding with a Redis cache (TTL EMBED_CACHE_TTL_SEC, default 1 day). Redis trouble never breaks
    search: on any error we just compute the embedding."""
    key = cache_key(text)
    try:
        r = client or _redis()
        raw = r.get(key)
        if raw and len(raw) == 4 * settings.EMBED_DIM:
            return np.frombuffer(raw, dtype=np.float32).tolist()
    except redis.RedisError:
        log.warning("embedding cache unavailable (read)")
        r = None
    vec = embed_batch([text])[0]
    if r is not None:
        try:
            r.set(key, np.asarray(vec, dtype=np.float32).tobytes(), ex=settings.EMBED_CACHE_TTL_SEC)
        except redis.RedisError:
            log.warning("embedding cache unavailable (write)")
    return vec
