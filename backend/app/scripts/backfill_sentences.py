"""Compute highlight sentence embeddings for chunks that have none, and rebuild rows that are stale (the sentence
splitter changed since they were stored). Safe to re-run: chunks whose stored sentences already match the current
splitter are skipped. No transcription: a few seconds per video.

Usage: docker compose exec api python -m app.scripts.backfill_sentences
"""
from collections import defaultdict

from sqlalchemy import delete, select

from app.core.db import SessionLocal
from app.models import Chunk, ChunkSentence
from app.services import sentences
from app.services.search_logic import split_sentences

BATCH = 100  # chunks per model call / commit


def main() -> int:
    db = SessionLocal()
    try:
        stored = defaultdict(list)
        for chunk_id, text in db.execute(select(ChunkSentence.chunk_id, ChunkSentence.text)
                                         .order_by(ChunkSentence.chunk_id, ChunkSentence.idx)):
            stored[chunk_id].append(text)
        todo, stale = [], 0
        for chunk_id, text in db.execute(select(Chunk.id, Chunk.text).order_by(Chunk.video_id, Chunk.idx)):
            expected = split_sentences(text)
            expected = expected if len(expected) > 1 else []  # single-sentence chunks need no rows
            if stored.get(chunk_id, []) != expected:
                todo.append((chunk_id, text))
                stale += chunk_id in stored
        total = 0
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            rows = sentences.build_rows(batch)
            db.execute(delete(ChunkSentence).where(ChunkSentence.chunk_id.in_([c for c, _ in batch])))
            db.add_all(rows)
            db.commit()
            total += len(rows)
            print(f"chunks {min(i + BATCH, len(todo))}/{len(todo)}: {total} sentence rows so far")
        print(f"Done: {len(todo)} chunk(s) needed work ({stale} had stale rows), inserted {total} sentence row(s).")
        return total
    finally:
        db.close()


if __name__ == "__main__":
    main()
