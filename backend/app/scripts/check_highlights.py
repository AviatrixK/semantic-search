"""Read-only: are search highlights served from the stored `chunk_sentences` vectors, or would search embed sentences per request?

For every chunk with more than one sentence it checks that stored rows exist and still equal what the current splitter
produces (exactly the test `retrieval._add_highlights` applies). Exit code 1 if any chunk would take the slow path.

Usage: docker compose exec api python -m app.scripts.check_highlights
Fix:   docker compose exec api python -m app.scripts.backfill_sentences   (then restart the worker if the splitter changed)
"""
import sys
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.exc import ProgrammingError

from app.core.db import SessionLocal
from app.models import Chunk, ChunkSentence, Video
from app.services.search_logic import split_sentences


def classify(chunk_text: str, stored: list[str]) -> str:
    """"single" (no rows needed), "stored" (fast path), "missing" or "stale" (both fall back to embedding on the fly)."""
    expected = split_sentences(chunk_text)
    if len(expected) <= 1:
        return "single"
    if not stored:
        return "missing"
    return "stored" if stored == expected else "stale"


def main() -> int:
    db = SessionLocal()
    try:
        try:
            stored = defaultdict(list)
            for chunk_id, text in db.execute(select(ChunkSentence.chunk_id, ChunkSentence.text)
                                             .order_by(ChunkSentence.chunk_id, ChunkSentence.idx)):
                stored[chunk_id].append(text)
            n_rows = db.scalar(select(func.count()).select_from(ChunkSentence))
        except ProgrammingError:
            print("The chunk_sentences table does not exist: every search embeds sentences on the fly. Apply the CREATE TABLE "
                  "from db/init.sql, then run backfill_sentences.")
            return 1
        per_video = defaultdict(lambda: defaultdict(int))
        for chunk_id, video_id, text in db.execute(select(Chunk.id, Chunk.video_id, Chunk.text)):
            per_video[video_id][classify(text, stored.get(chunk_id, []))] += 1
        titles = dict(db.execute(select(Video.id, Video.title)))
        totals = defaultdict(int)
        for video_id, counts in per_video.items():
            print(f"{titles.get(video_id, video_id)!s:50.50}  " + "  ".join(f"{k}={counts[k]}" for k in ("stored", "single", "missing", "stale")))
            for k, v in counts.items():
                totals[k] += v
        print(f"\nchunk_sentences rows: {n_rows}")
        print("TOTAL  " + "  ".join(f"{k}={totals[k]}" for k in ("stored", "single", "missing", "stale")))
        slow = totals["missing"] + totals["stale"]
        print("OK: search reads stored sentence vectors only." if not slow else
              f"PROBLEM: {slow} chunk(s) fall back to embedding sentences per request. Run backfill_sentences.")
        return 1 if slow else 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
