from dataclasses import dataclass

from app.core.config import settings


@dataclass
class ChunkSpan:
    start: float
    end: float
    text: str


def window(segments: list[dict], size: float | None = None, overlap: float | None = None) -> list[ChunkSpan]:
    """Group Whisper segments into ~`size`-second windows. Each new window starts at the first
    segment ending within the last `overlap` seconds of the previous one, so sentences at
    boundaries appear in both chunks. Segments are never split mid-sentence, so a single segment
    longer than `size` becomes its own (longer) chunk. Defaults come from settings."""
    size = settings.CHUNK_SECONDS if size is None else size
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size (got size={size}, overlap={overlap})")

    # Normalise: strip text, drop empty/whitespace-only segments, keep time order. Input is not mutated.
    segs = [{**s, "text": s["text"].strip()} for s in segments if s["text"].strip()]
    segs.sort(key=lambda s: (s["start"], s["end"]))

    chunks: list[ChunkSpan] = []
    i, n = 0, len(segs)
    while i < n:
        start = segs[i]["start"]
        j = i
        while j < n and (segs[j]["end"] - start <= size or j == i):
            j += 1
        chunks.append(ChunkSpan(start, segs[j - 1]["end"], " ".join(s["text"] for s in segs[i:j])))
        if j >= n:
            break
        boundary = segs[j - 1]["end"] - overlap
        nxt = j
        while nxt - 1 > i and segs[nxt - 1]["end"] > boundary:
            nxt -= 1
        i = nxt
    return chunks
