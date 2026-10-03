import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, time, timezone

from sqlalchemy import select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Chunk, ChunkSentence, Video
from app.services.embedding import embed_batch, embed_query
from app.services.search_logic import apply_threshold, best_index, dedupe_overlapping, split_sentences

log = logging.getLogger(__name__)
OVERFETCH = 3  # candidates per requested hit: thresholding and dedupe discard some
MAX_CANDIDATES = 200


def _stored_sentences(db: Session, chunk_ids: list) -> dict:
    """chunk_id -> [(sentence_text, vector)] in sentence order, from the chunk_sentences table."""
    try:
        with db.begin_nested():  # a missing table must not poison the surrounding transaction
            rows = db.execute(select(ChunkSentence.chunk_id, ChunkSentence.text, ChunkSentence.embedding)
                              .where(ChunkSentence.chunk_id.in_(chunk_ids))
                              .order_by(ChunkSentence.chunk_id, ChunkSentence.idx)).all()
    except ProgrammingError:
        log.warning("chunk_sentences table missing: apply the ALTER from the README; embedding sentences on the fly")
        return {}
    out: dict = defaultdict(list)
    for r in rows:
        out[r.chunk_id].append((r.text, r.embedding))
    return out


def _add_highlights(db: Session, hits: list[dict], query_vec: list[float]) -> None:
    """Sets hit["highlight"] to the sentence of the chunk closest to the query (cosine). Sentence vectors come from
    chunk_sentences (computed at ingest). Chunks without usable stored sentences (ingested before the table
    existed and not backfilled) fall back to embedding their sentences now, all in one batch: correct but ~10x
    slower (about 500 ms for k=10 on CPU)."""
    per_hit = [split_sentences(h["text"]) or [h["text"]] for h in hits]
    multi = [i for i, sentences in enumerate(per_hit) if len(sentences) > 1]
    for i, h in enumerate(hits):
        h["highlight"] = per_hit[i][0] if len(per_hit[i]) == 1 else None  # one sentence: the chunk itself
    stored = _stored_sentences(db, [hits[i]["chunk_id"] for i in multi]) if multi else {}
    fallback = []
    for i in multi:
        rows = stored.get(hits[i]["chunk_id"])
        if rows and [t for t, _ in rows] == per_hit[i]:  # stored rows still match what we would split today
            hits[i]["highlight"] = per_hit[i][best_index(query_vec, [v for _, v in rows])]
        else:
            fallback.append(i)
    if fallback:
        vecs = embed_batch([s for i in fallback for s in per_hit[i]])
        pos = 0
        for i in fallback:
            n = len(per_hit[i])
            hits[i]["highlight"] = per_hit[i][best_index(query_vec, vecs[pos:pos + n])]
            pos += n


def vector_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                  uploaded_after: date | None = None, min_score: float | None = None,
                  highlight: bool = True) -> list[dict]:
    """Semantic search. Over-fetches candidates, drops weak hits, dedupes overlapping chunks, keeps the best k.
    `uploaded_after`: only videos uploaded on or after that date (00:00 UTC)."""
    min_score = settings.MIN_SCORE if min_score is None else min_score
    qv = embed_query(query)
    limit = min(k * OVERFETCH, MAX_CANDIDATES)
    dist = Chunk.embedding.cosine_distance(qv)
    stmt = (select(Chunk.id.label("chunk_id"), Chunk.video_id, Video.title, Chunk.start_sec, Chunk.end_sec, Chunk.text, dist.label("dist"))
            .join(Video, Video.id == Chunk.video_id).order_by(dist).limit(limit))
    if video_id:
        stmt = stmt.where(Chunk.video_id == video_id)
    if uploaded_after:
        stmt = stmt.where(Video.created_at >= datetime.combine(uploaded_after, time.min, tzinfo=timezone.utc))
    # HNSW returns at most ef_search rows (default 40) however large LIMIT is, and filters apply after the index scan.
    db.execute(text(f"SET LOCAL hnsw.ef_search = {min(1000, max(100, limit * 2))}"))
    hits = [{"chunk_id": r.chunk_id, "video_id": str(r.video_id), "title": r.title, "start_sec": r.start_sec, "end_sec": r.end_sec,
             "text": r.text, "score": round(1 - r.dist, 4)} for r in db.execute(stmt)]
    hits = dedupe_overlapping(apply_threshold(hits, min_score))[:k]
    if highlight and hits:
        _add_highlights(db, hits, qv)
    else:
        for h in hits:
            h["highlight"] = None
    return hits

# Week 10: add keyword_search() using chunks.tsv + reciprocal rank fusion -> hybrid_search().


def list_chunks(db: Session, video_id: uuid.UUID) -> list[dict]:
    rows = db.scalars(select(Chunk).where(Chunk.video_id == video_id).order_by(Chunk.idx)).all()
    return [{"idx": c.idx, "start_sec": c.start_sec, "end_sec": c.end_sec, "text": c.text} for c in rows]


def log_query(db: Session, user_id: str, query: str, mode: str):
    db.execute(text("INSERT INTO search_logs (id, user_id, query, mode) VALUES (:i, :u, :q, :m)"),
               {"i": uuid.uuid4(), "u": uuid.UUID(user_id), "q": query, "m": mode})
    db.commit()
