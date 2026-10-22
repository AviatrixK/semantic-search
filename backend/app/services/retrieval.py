import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, time, timezone

from sqlalchemy import func, literal_column, or_, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Chunk, ChunkSentence, Video
from app.services import embedders, vectorstore
from app.services.embedding import embed_batch, embed_query
from app.services.search_logic import (apply_threshold, best_index, best_lexical_index, dedupe_overlapping, rrf_merge,
                                       split_sentences)
from app.services.timing import Timings, measure

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


def _add_highlights(db: Session, hits: list[dict], query_vec: list[float], timings: Timings | None = None) -> None:
    """Sets hit["highlight"] to the sentence of the chunk closest to the query (cosine). Sentence vectors come from
    chunk_sentences (computed at ingest). Chunks without usable stored sentences (ingested before the table
    existed and not backfilled) fall back to embedding their sentences now, all in one batch: correct but ~10x
    slower (about 500 ms for k=10 on CPU). LOCAL embedder only: see _lexical_highlights for the others."""
    per_hit = [split_sentences(h["text"]) or [h["text"]] for h in hits]
    multi = [i for i, sentences in enumerate(per_hit) if len(sentences) > 1]
    for i, h in enumerate(hits):
        h["highlight"] = per_hit[i][0] if len(per_hit[i]) == 1 else None  # one sentence: the chunk itself
    with measure(timings, "hl_lookup"):
        stored = _stored_sentences(db, [hits[i]["chunk_id"] for i in multi]) if multi else {}
    fallback = []
    for i in multi:
        rows = stored.get(hits[i]["chunk_id"])
        if rows and [t for t, _ in rows] == per_hit[i]:  # stored rows still match what we would split today
            hits[i]["highlight"] = per_hit[i][best_index(query_vec, [v for _, v in rows])]
        else:
            fallback.append(i)
    if timings is not None:
        timings.count("single", len(hits) - len(multi))
        timings.count("stored", len(multi) - len(fallback))
        timings.count("fallback", len(fallback))
    if fallback:
        log.warning("search highlight: %d of %d chunk(s) have no usable chunk_sentences rows, embedding their sentences "
                    "now (slow). Run: python -m app.scripts.backfill_sentences", len(fallback), len(hits))
        with measure(timings, "hl_embed"):
            vecs = embed_batch([s for i in fallback for s in per_hit[i]])
        pos = 0
        for i in fallback:
            n = len(per_hit[i])
            hits[i]["highlight"] = per_hit[i][best_index(query_vec, vecs[pos:pos + n])]
            pos += n


def _lexical_highlights(hits: list[dict], query: str, timings: Timings | None = None) -> None:
    """Highlight without any model: the sentence sharing the most words with the query. Used for hosted embedding providers (they
    store no per-sentence vectors) and whenever the query embedding is unavailable."""
    for h in hits:
        sentences = split_sentences(h["text"]) or [h["text"]]
        i = 0 if len(sentences) == 1 else best_lexical_index(query, sentences)
        h["highlight"] = sentences[i] if i is not None else None
    if timings is not None:
        timings.count("lexical", len(hits))


def _filters(stmt, video_id, uploaded_after):
    if video_id:
        stmt = stmt.where(Chunk.video_id == video_id)
    if uploaded_after:
        stmt = stmt.where(Video.created_at >= datetime.combine(uploaded_after, time.min, tzinfo=timezone.utc))
    return stmt


def _row_to_hit(r, score: float) -> dict:
    return {"chunk_id": r.chunk_id, "video_id": str(r.video_id), "title": r.title, "start_sec": r.start_sec,
            "end_sec": r.end_sec, "text": r.text, "score": round(score, 4)}


def _query_vec(embedder, query: str, timings: Timings | None = None) -> list[float] | None:
    """The query's vector, or None when a hosted provider cannot be reached in time (EMBED_QUERY_TIMEOUT_SEC, one retry): the
    search then runs on keywords alone, and says so in the log (WARNING) and in the X-Search-Fallback response header."""
    try:
        with measure(timings, "embed"):
            if embedder.uses_legacy_column:
                return embed_query(query)  # the local model (cached in Redis like every query vector)
            return embedders.query_vector(embedder, query)
    except embedders.EmbeddingError as exc:
        log.warning("query embedding failed (%s: %s): falling back to keyword-only search", type(exc).__name__, exc)
        if timings is not None:
            timings.count("embed_failed")
        return None


def _vector_candidates(db: Session, qv: list[float], limit: int, video_id, uploaded_after,
                       timings: Timings | None = None, embedder=None) -> list[dict]:
    """The `limit` chunks closest to the query vector, best first, score = cosine similarity. With a hosted provider each chunk
    is matched through the better of its native-text and English-text vectors."""
    embedder = embedder or embedders.get_embedder()
    if not embedder.uses_legacy_column:
        with measure(timings, "vector_sql"):
            return vectorstore.nearest(db, embedder, qv, limit, video_id, uploaded_after)
    dist = Chunk.embedding.cosine_distance(qv)
    stmt = _filters(select(Chunk.id.label("chunk_id"), Chunk.video_id, Video.title, Chunk.start_sec, Chunk.end_sec, Chunk.text,
                           dist.label("dist")).join(Video, Video.id == Chunk.video_id).where(Chunk.embedding.is_not(None))
                    .order_by(dist).limit(limit), video_id, uploaded_after)
    # HNSW returns at most ef_search rows (default 40) however large LIMIT is, and filters apply after the index scan.
    with measure(timings, "vector_sql"):
        db.execute(text(f"SET LOCAL hnsw.ef_search = {min(1000, max(100, limit * 2))}"))
        return [_row_to_hit(r, 1 - r.dist) for r in db.execute(stmt)]


def _finish(db: Session, hits: list[dict], qv: list[float] | None, highlight: bool, timings: Timings | None = None,
            *, query: str = "", embedder=None) -> list[dict]:
    embedder = embedder or embedders.get_embedder()
    if not highlight or not hits:
        for h in hits:
            h["highlight"] = None
    elif qv is not None and embedder.supports_sentence_vectors:
        _add_highlights(db, hits, qv, timings)
    else:
        _lexical_highlights(hits, query, timings)
    return hits


def vector_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                  uploaded_after: date | None = None, min_score: float | None = None,
                  highlight: bool = True, timings: Timings | None = None, embedder=None) -> list[dict]:
    """Semantic search. Over-fetches candidates, drops weak hits, dedupes overlapping chunks, keeps the best k.
    `uploaded_after`: only videos uploaded on or after that date (00:00 UTC). If the query cannot be embedded (hosted provider
    down or slow) it answers with a keyword search instead."""
    embedder = embedder or embedders.get_embedder()
    min_score = settings.min_score_for(embedder.provider) if min_score is None else min_score
    qv = _query_vec(embedder, query, timings)
    if qv is None:
        return keyword_search(db, query, k, video_id, uploaded_after, highlight, timings, embedder=embedder)
    hits = _vector_candidates(db, qv, min(k * OVERFETCH, MAX_CANDIDATES), video_id, uploaded_after, timings, embedder)
    hits = dedupe_overlapping(apply_threshold(hits, min_score))[:k]
    return _finish(db, hits, qv, highlight, timings, query=query, embedder=embedder)


def _keyword_candidates(db: Session, query: str, limit: int, video_id, uploaded_after,
                        qv: list[float] | None, timings: Timings | None = None, embedder=None) -> list[dict]:
    """Chunks matching the query in EITHER keyword index, best first. Two indexes, two tsqueries: `tsv` ('simple': no stemming, no
    stop words, any script) over the native text and its romanisation, and `tsv_en` ('english': stemming, so "speaking" finds "speak")
    over the English text. A chunk found by both ranks higher (the two ts_rank_cd values are added). websearch_to_tsquery understands
    plain text, "quoted phrases", OR and -exclusions and never raises on odd input.
    score = cosine similarity when the query vector is known (hybrid), otherwise rank / (1 + rank), a 0..1 keyword relevance."""
    embedder = embedder or embedders.get_embedder()
    tsq = func.websearch_to_tsquery("simple", query)
    tsq_en = func.websearch_to_tsquery("english", query)
    tsv, tsv_en = literal_column("chunks.tsv"), literal_column("chunks.tsv_en")
    rank = func.ts_rank_cd(tsv, tsq) + func.ts_rank_cd(tsv_en, tsq_en)
    cols = [Chunk.id.label("chunk_id"), Chunk.video_id, Video.title, Chunk.start_sec, Chunk.end_sec, Chunk.text, rank.label("rank")]
    legacy_cosine = qv is not None and embedder.uses_legacy_column
    if legacy_cosine:
        cols.append(Chunk.embedding.cosine_distance(qv).label("dist"))
    stmt = _filters(select(*cols).join(Video, Video.id == Chunk.video_id).where(or_(tsv.op("@@")(tsq), tsv_en.op("@@")(tsq_en)))
                    .order_by(rank.desc(), Chunk.video_id, Chunk.start_sec).limit(limit), video_id, uploaded_after)
    with measure(timings, "keyword_sql"):
        rows = list(db.execute(stmt))
        cosine: dict = {}
        if qv is not None and not embedder.uses_legacy_column and rows:
            cosine = vectorstore.score_chunks(db, embedder, qv, [r.chunk_id for r in rows])
    out = []
    for r in rows:
        if legacy_cosine:
            score = 1 - r.dist if r.dist is not None else 0.0
        elif qv is not None:
            score = cosine.get(r.chunk_id, 0.0)
        else:
            score = r.rank / (1 + r.rank)
        out.append(_row_to_hit(r, score))
    return out


def keyword_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                   uploaded_after: date | None = None, highlight: bool = True,
                   timings: Timings | None = None, embedder=None) -> list[dict]:
    """Full-text search: exact words, names and acronyms in the native and romanised text, plus stemmed matches in the English text. Overlapping chunks are
    collapsed like in vector search. With the local model, highlights embed the sentences of the k hits (`highlight=False` keeps
    this mode model-free); hosted providers are never called here: highlights are lexical."""
    embedder = embedder or embedders.get_embedder()
    hits = dedupe_overlapping(_keyword_candidates(db, query, min(k * OVERFETCH, MAX_CANDIDATES), video_id, uploaded_after,
                                                  None, timings, embedder))[:k]
    qv = _query_vec(embedder, query, timings) if highlight and hits and embedder.supports_sentence_vectors else None
    return _finish(db, hits, qv, highlight, timings, query=query, embedder=embedder)


def hybrid_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None,
                  uploaded_after: date | None = None, min_score: float | None = None,
                  highlight: bool = True, rerank: bool | None = None, timings: Timings | None = None,
                  embedder=None) -> list[dict]:
    """Vector + keyword search merged with Reciprocal Rank Fusion. Each side contributes its top HYBRID_CANDIDATES
    (vector candidates below MIN_SCORE are dropped first: a weak neighbour should not collect rank credit; keyword
    candidates are kept, an exact word match is evidence on its own). Overlapping chunks collapse to the best fused
    one; with RERANK on, the top RERANK_CANDIDATES are rescored by a cross-encoder. `score` stays the cosine
    similarity (what the UI shows), `rrf` is the fused ranking score, `found_by` says which searches returned it.
    If the query cannot be embedded in time (hosted provider down or slow), only the keyword side runs."""
    use_rerank = settings.RERANK if rerank is None else rerank
    embedder = embedder or embedders.get_embedder()
    min_score = settings.min_score_for(embedder.provider) if min_score is None else min_score
    qv = _query_vec(embedder, query, timings)
    n = max(settings.HYBRID_CANDIDATES, k)
    if qv is None:  # keyword-only fallback: already ranked, nothing to fuse
        kw = dedupe_overlapping(_keyword_candidates(db, query, n, video_id, uploaded_after, None, timings, embedder))[:k]
        for h in kw:
            h["found_by"] = ["keyword"]
        return _finish(db, kw, None, highlight, timings, query=query, embedder=embedder)
    vec = apply_threshold(_vector_candidates(db, qv, n, video_id, uploaded_after, timings, embedder), min_score)
    kw = _keyword_candidates(db, query, n, video_id, uploaded_after, qv, timings, embedder)
    merged = dedupe_overlapping(rrf_merge({"vector": vec, "keyword": kw}, k=settings.RRF_K), key="rrf")
    if use_rerank:
        from app.services import rerank as reranker
        with measure(timings, "rerank"):
            hits = reranker.rerank(query, merged, k)
    else:
        hits = merged[:k]
    return _finish(db, hits, qv, highlight, timings, query=query, embedder=embedder)


def list_chunks(db: Session, video_id: uuid.UUID) -> list[dict]:
    rows = db.scalars(select(Chunk).where(Chunk.video_id == video_id).order_by(Chunk.idx)).all()
    return [{"idx": c.idx, "start_sec": c.start_sec, "end_sec": c.end_sec, "text": c.text} for c in rows]


def log_query(db: Session, user_id: str, query: str, mode: str):
    db.execute(text("INSERT INTO search_logs (id, user_id, query, mode) VALUES (:i, :u, :q, :m)"),
               {"i": uuid.uuid4(), "u": uuid.UUID(user_id), "q": query, "m": mode})
    db.commit()
