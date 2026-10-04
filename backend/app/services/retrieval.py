import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, time, timezone

from sqlalchemy import func, literal_column, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Chunk, ChunkSentence, Video
from app.services.embedding import embed_batch, embed_query
from app.services.search_logic import apply_threshold, best_index, dedupe_overlapping, rrf_merge, split_sentences

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


def _filters(stmt, video_id, uploaded_after):
    if video_id:
        stmt = stmt.where(Chunk.video_id == video_id)
    if uploaded_after:
        stmt = stmt.where(Video.created_at >= datetime.combine(uploaded_after, time.min, tzinfo=timezone.utc))
    return stmt


def _row_to_hit(r, score: float) -> dict:
    return {"chunk_id": r.chunk_id, "video_id": str(r.video_id), "title": r.title, "start_sec": r.start_sec,
            "end_sec": r.end_sec, "text": r.text, "score": round(score, 4)}


def _vector_candidates(db: Session, qv: list[float], limit: int, video_id, uploaded_after) -> list[dict]:
    """The `limit` chunks closest to the query vector, best first, score = cosine similarity."""
    dist = Chunk.embedding.cosine_distance(qv)
    stmt = _filters(select(Chunk.id.label("chunk_id"), Chunk.video_id, Video.title, Chunk.start_sec, Chunk.end_sec, Chunk.text,
                           dist.label("dist")).join(Video, Video.id == Chunk.video_id).order_by(dist).limit(limit),
                    video_id, uploaded_after)
    # HNSW returns at most ef_search rows (default 40) however large LIMIT is, and filters apply after the index scan.
    db.execute(text(f"SET LOCAL hnsw.ef_search = {min(1000, max(100, limit * 2))}"))
    return [_row_to_hit(r, 1 - r.dist) for r in db.execute(stmt)]


def _finish(db: Session, hits: list[dict], qv: list[float] | None, highlight: bool) -> list[dict]:
    if highlight and hits and qv is not None:
        _add_highlights(db, hits, qv)
    else:
        for h in hits:
            h["highlight"] = None
    return hits


def vector_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                  uploaded_after: date | None = None, min_score: float | None = None,
                  highlight: bool = True) -> list[dict]:
    """Semantic search. Over-fetches candidates, drops weak hits, dedupes overlapping chunks, keeps the best k.
    `uploaded_after`: only videos uploaded on or after that date (00:00 UTC)."""
    min_score = settings.MIN_SCORE if min_score is None else min_score
    qv = embed_query(query)
    hits = _vector_candidates(db, qv, min(k * OVERFETCH, MAX_CANDIDATES), video_id, uploaded_after)
    hits = dedupe_overlapping(apply_threshold(hits, min_score))[:k]
    return _finish(db, hits, qv, highlight)


def _keyword_candidates(db: Session, query: str, limit: int, video_id, uploaded_after,
                        qv: list[float] | None) -> list[dict]:
    """Chunks whose `tsv` matches the query, best ts_rank_cd first. websearch_to_tsquery understands plain text,
    "quoted phrases", OR and -exclusions and never raises on odd input; a query of only stop words matches nothing.
    score = cosine similarity when the query vector is known (hybrid), otherwise rank / (1 + rank), a 0..1 keyword
    relevance."""
    tsq = func.websearch_to_tsquery("english", query)
    tsv = literal_column("chunks.tsv")
    rank = func.ts_rank_cd(tsv, tsq)
    cols = [Chunk.id.label("chunk_id"), Chunk.video_id, Video.title, Chunk.start_sec, Chunk.end_sec, Chunk.text, rank.label("rank")]
    if qv is not None:
        cols.append(Chunk.embedding.cosine_distance(qv).label("dist"))
    stmt = _filters(select(*cols).join(Video, Video.id == Chunk.video_id).where(tsv.op("@@")(tsq))
                    .order_by(rank.desc(), Chunk.video_id, Chunk.start_sec).limit(limit), video_id, uploaded_after)
    return [_row_to_hit(r, 1 - r.dist if qv is not None else r.rank / (1 + r.rank)) for r in db.execute(stmt)]


def keyword_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                   uploaded_after: date | None = None, highlight: bool = True) -> list[dict]:
    """Full-text search (no model, so no embedding cost): exact words, names and acronyms, with English stemming.
    Overlapping chunks are collapsed like in vector search. Highlights embed the sentences of the k hits, so
    `highlight=False` keeps this mode model-free."""
    hits = dedupe_overlapping(_keyword_candidates(db, query, min(k * OVERFETCH, MAX_CANDIDATES), video_id, uploaded_after, None))[:k]
    return _finish(db, hits, embed_query(query) if highlight and hits else None, highlight)


def hybrid_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                  uploaded_after: date | None = None, min_score: float | None = None,
                  highlight: bool = True, rerank: bool | None = None) -> list[dict]:
    """Vector + keyword search merged with Reciprocal Rank Fusion. Each side contributes its top HYBRID_CANDIDATES
    (vector candidates below MIN_SCORE are dropped first: a weak neighbour should not collect rank credit; keyword
    candidates are kept, an exact word match is evidence on its own). Overlapping chunks collapse to the best fused
    one; with RERANK on, the top RERANK_CANDIDATES are rescored by a cross-encoder. `score` stays the cosine
    similarity (what the UI shows), `rrf` is the fused ranking score, `found_by` says which searches returned it."""
    min_score = settings.MIN_SCORE if min_score is None else min_score
    use_rerank = settings.RERANK if rerank is None else rerank
    qv = embed_query(query)
    n = max(settings.HYBRID_CANDIDATES, k)
    vec = apply_threshold(_vector_candidates(db, qv, n, video_id, uploaded_after), min_score)
    kw = _keyword_candidates(db, query, n, video_id, uploaded_after, qv)
    merged = dedupe_overlapping(rrf_merge({"vector": vec, "keyword": kw}, k=settings.RRF_K), key="rrf")
    if use_rerank:
        from app.services import rerank as reranker
        hits = reranker.rerank(query, merged, k)
    else:
        hits = merged[:k]
    return _finish(db, hits, qv, highlight)


def list_chunks(db: Session, video_id: uuid.UUID) -> list[dict]:
    rows = db.scalars(select(Chunk).where(Chunk.video_id == video_id).order_by(Chunk.idx)).all()
    return [{"idx": c.idx, "start_sec": c.start_sec, "end_sec": c.end_sec, "text": c.text} for c in rows]


def log_query(db: Session, user_id: str, query: str, mode: str):
    db.execute(text("INSERT INTO search_logs (id, user_id, query, mode) VALUES (:i, :u, :q, :m)"),
               {"i": uuid.uuid4(), "u": uuid.UUID(user_id), "q": query, "m": mode})
    db.commit()
