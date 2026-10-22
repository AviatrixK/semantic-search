import logging
import time
from functools import lru_cache

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
    log.info("worker: torch threads limited to %d", torch.get_num_threads())


_warm = False
_threads_at_warmup: int | None = None


def warm_threads() -> int | None:
    """torch's CPU thread count as it was when warm-up finished (for the startup log); None before or without torch."""
    return _threads_at_warmup


def warm_up() -> float:
    """Loads the embedding model and runs one encode, so the first real query pays neither the model load (seconds) nor the
    first-call initialisation. Returns the milliseconds it took. Called by the API at startup, in every gunicorn worker.
    Deliberately does NOT touch thread limits: those belong to the Celery worker (see limit_torch_threads)."""
    global _warm, _threads_at_warmup
    started = time.perf_counter()
    embed_batch(["warm up"])  # _model() loads on first use; encoding initialises the tokenizer and thread pools
    try:
        import torch
        _threads_at_warmup = torch.get_num_threads()
    except Exception:  # only for the log line
        _threads_at_warmup = None
    _warm = True
    return (time.perf_counter() - started) * 1000


def is_warm() -> bool:
    return _warm


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, batch_size=64, normalize_embeddings=True).tolist()


def cache_key(query: str) -> str:
    """Cache key of a query embedded by the LOCAL model (see embedders.cache_key: provider + model + dimensions + query)."""
    from app.services import embedders
    return embedders.cache_key(embedders.LocalEmbedder(), query)


def embed_query(text: str, *, client=None) -> list[float]:
    """Query embedding by the LOCAL model, cached in Redis. Search uses embedders.query_vector with the configured provider."""
    from app.services import embedders
    return embedders.query_vector(embedders.LocalEmbedder(), text, client=client)
