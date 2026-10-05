"""Reading a stretch of a video's transcript. Chunks overlap by several seconds (the overlap is whole segments, so a chunk
starts with exactly the text its predecessor ends with); stitching removes that repeat so a window reads once, in order."""
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Video


def trim_overlap(prev: str, nxt: str) -> tuple[str, int]:
    """`nxt` without the start that repeats the end of `prev` (longest match on word boundaries), and how much was removed."""
    for length in range(min(len(prev), len(nxt)), 0, -1):
        if length < len(nxt) and nxt[length] != " ":
            continue  # must end on a word boundary in `nxt`
        start = len(prev) - length
        if start > 0 and prev[start - 1] != " ":
            continue  # and begin on one in `prev`
        if prev.endswith(nxt[:length]):
            return nxt[length:].lstrip(), length
    return nxt, 0


def stitch_chunks(chunks: list[dict]) -> list[dict]:
    """Chunks (idx, start_sec, end_sec, text) -> non-repeating segments (start_sec, end_sec, text). A chunk's repeated start is
    dropped only when it also overlaps the previous chunk in time; the new material begins where the previous chunk ended."""
    out: list[dict] = []
    prev: dict | None = None
    for c in sorted(chunks, key=lambda c: c["idx"]):
        text = " ".join(c["text"].split())
        start = c["start_sec"]
        if prev is not None and c["start_sec"] < prev["end_sec"]:
            trimmed, removed = trim_overlap(" ".join(prev["text"].split()), text)
            if removed:
                text, start = trimmed, max(prev["end_sec"], c["start_sec"])
        prev = c
        if text:
            out.append({"start_sec": start, "end_sec": c["end_sec"], "text": text})
    return out


def get_window(db: Session, video_id: uuid.UUID, start_sec: float, end_sec: float) -> dict | None:
    """{video_id, title, segments: [...]} for the chunks that overlap [start_sec, end_sec], or None if the video does not exist."""
    video = db.get(Video, video_id)
    if video is None:
        return None
    rows = db.execute(select(Chunk.idx, Chunk.start_sec, Chunk.end_sec, Chunk.text)
                      .where(Chunk.video_id == video_id, Chunk.end_sec > start_sec, Chunk.start_sec < end_sec)
                      .order_by(Chunk.idx)).all()
    chunks = [{"idx": r.idx, "start_sec": r.start_sec, "end_sec": r.end_sec, "text": r.text} for r in rows]
    return {"video_id": str(video_id), "title": video.title, "segments": stitch_chunks(chunks)}
