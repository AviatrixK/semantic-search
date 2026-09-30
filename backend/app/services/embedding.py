from functools import lru_cache

from app.core.config import settings


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(settings.EMBED_MODEL)
    dim = m.get_sentence_embedding_dimension()
    if dim != settings.EMBED_DIM:
        raise RuntimeError(f"EMBED_DIM={settings.EMBED_DIM} but model outputs {dim}; fix .env and schema")
    return m


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, batch_size=32, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_batch([text])[0]
