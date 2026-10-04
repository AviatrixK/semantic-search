"""Optional cross-encoder reranking (RERANK=true). The model is only loaded, once per process, when it is first used,
so with RERANK=false nothing is downloaded and no memory is spent."""
import logging
from functools import lru_cache

from app.core.config import settings
from app.services.search_logic import rerank_order

log = logging.getLogger(__name__)


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.RERANK_MODEL)


def score_pairs(query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _model().predict([(query, t) for t in texts], batch_size=32)]


def rerank(query: str, hits: list[dict], k: int, scorer=None) -> list[dict]:
    """Rescores the first RERANK_CANDIDATES hits with the cross-encoder and returns the best k. Any failure
    (model missing, download blocked, out of memory) logs and falls back to the incoming order, so search never breaks
    because the optional reranker did."""
    pool = hits[:settings.RERANK_CANDIDATES]
    if len(pool) < 2:
        return hits[:k]
    try:
        scores = (scorer or score_pairs)(query, [h["text"] for h in pool])
        return rerank_order(pool, scores)[:k]
    except Exception:
        log.exception("reranker failed; keeping the hybrid order")
        return hits[:k]
