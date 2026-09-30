from dataclasses import dataclass


@dataclass
class ChunkSpan:
    start: float
    end: float
    text: str


def window(segments: list[dict], size: float = 30.0, overlap: float = 5.0) -> list[ChunkSpan]:
    """Group Whisper segments into ~`size`-second windows. Each new window starts at the first
    segment beginning within the last `overlap` seconds of the previous one, so sentences at
    boundaries appear in both chunks. Segments are never split mid-sentence."""
    chunks: list[ChunkSpan] = []
    i, n = 0, len(segments)
    while i < n:
        start = segments[i]["start"]
        j = i
        while j < n and (segments[j]["end"] - start <= size or j == i):
            j += 1
        chunks.append(ChunkSpan(start, segments[j - 1]["end"], " ".join(s["text"] for s in segments[i:j])))
        if j >= n:
            break
        boundary = segments[j - 1]["end"] - overlap
        nxt = j
        while nxt - 1 > i and segments[nxt - 1]["end"] > boundary:
            nxt -= 1
        i = nxt
    return chunks
