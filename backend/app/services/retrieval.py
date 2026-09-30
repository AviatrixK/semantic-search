import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Video
from app.services.embedding import embed_query


def vector_search(db: Session, query: str, k: int = 10, video_id: uuid.UUID | None = None) -> list[dict]:
    qv = embed_query(query)
    dist = Chunk.embedding.cosine_distance(qv)
    stmt = (select(Chunk, Video.title, dist.label("dist")).join(Video, Video.id == Chunk.video_id)
            .order_by(dist).limit(k))
    if video_id:
        stmt = stmt.where(Chunk.video_id == video_id)
    return [{"video_id": str(c.video_id), "title": title, "start_sec": c.start_sec, "end_sec": c.end_sec,
             "text": c.text, "score": round(1 - d, 4)} for c, title, d in db.execute(stmt)]

# Week 10: add keyword_search() using chunks.tsv + reciprocal rank fusion -> hybrid_search().


def log_query(db: Session, user_id: str, query: str, mode: str):
    from sqlalchemy import text
    db.execute(text("INSERT INTO search_logs (id, user_id, query, mode) VALUES (:i, :u, :q, :m)"),
               {"i": uuid.uuid4(), "u": user_id, "q": query, "m": mode})
    db.commit()
