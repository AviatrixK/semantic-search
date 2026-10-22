"""Vectors of the hosted embedding models, in the chunk_embeddings table.

One row per (chunk, model, text variant). The `embedding` column has no fixed size, so models of different dimensions live side by
side (that is what lets the eval compare providers on the same chunks). pgvector's HNSW index needs a fixed size, so every model gets
its own PARTIAL index on the cast expression:

    CREATE INDEX chunk_emb_hnsw_<hash> ON chunk_embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WHERE model = 'gemini/gemini-embedding-2@1536'

and queries must repeat exactly that expression and predicate to use it (see `nearest`). HNSW supports at most 2000 dimensions,
which is why EMBED_DIMENSIONS is capped there.

Text variants ("fields"): native = the spoken text in its own script, en = the English translation. Search takes the better of the
two per chunk."""
import hashlib
import re
import uuid
from datetime import date, datetime, time, timezone
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import Chunk, ChunkEmbedding

FIELDS = ("native", "en")
_SAFE = re.compile(r"^[A-Za-z0-9._:/@+\-]+$")  # a model key is put into SQL as a literal (an index predicate needs a constant)
OVERFETCH = 2  # every chunk can match through two rows (native and en): fetch more, then keep the best row per chunk


def _literal(embedder) -> str:
    key = embedder.key
    if not _SAFE.match(key):
        raise ValueError(f"unsafe model key {key!r}")
    return key


def index_name(embedder) -> str:
    return "chunk_emb_hnsw_" + hashlib.sha1(embedder.key.encode()).hexdigest()[:12]


def ensure_index(engine, embedder) -> None:
    """Creates the model's partial HNSW index if it is missing (idempotent). Runs outside a transaction (autocommit), so it can be
    called from the middle of a job. Building it on a table that already holds this model's vectors takes a moment."""
    if embedder.uses_legacy_column:
        return
    dim, key = int(embedder.dim), _literal(embedder)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name(embedder)} ON chunk_embeddings "
                          f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) WHERE model = '{key}'"))


def field_texts(chunk) -> dict[str, str]:
    """The texts of a chunk to embed, by field. Empty variants are skipped (nothing to embed)."""
    native = (chunk.text_native or chunk.text or "").strip()
    en = (chunk.text_en or "").strip()
    return {k: v for k, v in (("native", native), ("en", en)) if v}


def store(db: Session, embedder, rows: Iterable[tuple[uuid.UUID, str, list[float]]]) -> int:
    """Upserts (chunk_id, field, vector) rows for this model. Re-running with the same rows changes nothing (resumable)."""
    values = [{"chunk_id": cid, "model": embedder.key, "field": field, "embedding": vec} for cid, field, vec in rows]
    if not values:
        return 0
    stmt = pg_insert(ChunkEmbedding).values(values)
    db.execute(stmt.on_conflict_do_update(index_elements=["chunk_id", "model", "field"], set_={"embedding": stmt.excluded.embedding}))
    return len(values)


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


def nearest(db: Session, embedder, qv: list[float], limit: int, video_id=None, uploaded_after: date | None = None) -> list[dict]:
    """The `limit` chunks closest to the query vector by this model, best first. A chunk counts once, with its better text variant
    (`matched` says which). score = cosine similarity."""
    dim, key = int(embedder.dim), _literal(embedder)
    if len(qv) != dim:
        raise ValueError(f"query vector has {len(qv)} dimensions, model {key} has {dim}")
    expr = f"ce.embedding::vector({dim})"
    where = [f"ce.model = '{key}'"]
    params: dict = {"q": _vector_literal(qv), "lim": limit * OVERFETCH}
    if video_id:
        where.append("c.video_id = :vid")
        params["vid"] = video_id
    if uploaded_after:
        where.append("v.created_at >= :after")
        params["after"] = datetime.combine(uploaded_after, time.min, tzinfo=timezone.utc)
    sql = (f"SELECT c.id AS chunk_id, c.video_id, v.title, c.start_sec, c.end_sec, c.text, ce.field, "
           f"{expr} <=> CAST(:q AS vector({dim})) AS dist "
           f"FROM chunk_embeddings ce JOIN chunks c ON c.id = ce.chunk_id JOIN videos v ON v.id = c.video_id "
           f"WHERE {' AND '.join(where)} ORDER BY {expr} <=> CAST(:q AS vector({dim})) LIMIT :lim")
    # HNSW returns at most hnsw.ef_search rows (default 40) and applies WHERE filters after the scan: raise it to cover the limit.
    db.execute(text(f"SET LOCAL hnsw.ef_search = {min(1000, max(100, limit * OVERFETCH * 2))}"))
    hits: dict = {}
    for r in db.execute(text(sql), params):
        if r.chunk_id not in hits:  # rows arrive best first, so the first row of a chunk is its better variant
            hits[r.chunk_id] = {"chunk_id": r.chunk_id, "video_id": str(r.video_id), "title": r.title, "start_sec": r.start_sec,
                                "end_sec": r.end_sec, "text": r.text, "score": round(1 - r.dist, 4), "matched": r.field}
    return list(hits.values())[:limit]


def missing(db: Session, embedder, video_ids: list | None = None) -> list[tuple[uuid.UUID, str, str]]:
    """(chunk_id, field, text) for every text variant that has no vector for this model yet. This is what makes re-embedding
    resumable: run it again and only the gaps are filled."""
    q = db.query(Chunk)
    if video_ids:
        q = q.filter(Chunk.video_id.in_(video_ids))
    chunks = q.order_by(Chunk.video_id, Chunk.idx).all()
    have = {(cid, f) for cid, f in db.query(ChunkEmbedding.chunk_id, ChunkEmbedding.field)
            .filter(ChunkEmbedding.model == embedder.key).all()}
    return [(c.id, f, t) for c in chunks for f, t in field_texts(c).items() if (c.id, f) not in have]


def counts_by_model(db: Session) -> dict[str, int]:
    return dict(db.execute(text("SELECT model, count(*) FROM chunk_embeddings GROUP BY model")).all())


def score_chunks(db: Session, embedder, qv: list[float], chunk_ids: list) -> dict:
    """chunk_id -> cosine similarity to the query vector (the better of its text variants) for the given chunks. Used to give
    keyword hits the same `score` a vector hit would have, so hybrid results are comparable. Chunks without vectors are absent."""
    if not chunk_ids:
        return {}
    dim, key = int(embedder.dim), _literal(embedder)
    rows = db.execute(text(f"SELECT chunk_id, min(embedding::vector({dim}) <=> CAST(:q AS vector({dim}))) AS dist "
                           f"FROM chunk_embeddings WHERE model = '{key}' AND chunk_id = ANY(:ids) GROUP BY chunk_id"),
                      {"q": _vector_literal(qv), "ids": list(chunk_ids)})
    return {r.chunk_id: 1 - r.dist for r in rows}
