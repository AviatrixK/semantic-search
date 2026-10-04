"""Evidence tracking and citation checking. The agent may only cite what its tools actually returned: every [video_id@seconds]
in the final answer is checked against the clips and transcript segments the model was really shown. Anything else is
removed (a made-up video id, or a time outside every retrieved range) and the survivors become numbered [n] markers,
the same shape the RAG answers use, so the frontend chips work unchanged."""
import re
from dataclasses import dataclass, field

from app.services.rag import NO_CONTEXT_ANSWER, Citation

TOLERANCE_SEC = 1.0  # a ref is int(start): 283 for a clip starting at 283.4
UNFINISHED_PREFIX = "I could not finish my research"  # starts the fallback answer (also lets the eval recognise it)
MIN_PREFIX = 8  # models sometimes shorten a UUID; accept a unique prefix of at least this many characters

_REF = r"([A-Za-z0-9_-]{4,40})@(\d+(?:\.\d+)?)s?"
_GROUP = re.compile(rf"\[\s*{_REF}(?:\s*[,;]\s*{_REF})*\s*\]")
_ONE = re.compile(_REF)


@dataclass
class Evidence:
    """Every (video, time range) the model has seen in a tool result."""
    ranges: list[tuple[str, float, float, str]] = field(default_factory=list)  # (video_id, start, end, title)

    def add_result(self, result: dict) -> None:
        for key in ("clips", "segments"):
            for item in result.get(key, []) or []:
                vid = item.get("video_id") or result.get("video_id")
                if vid is None or "start_sec" not in item or "end_sec" not in item:
                    continue
                self.ranges.append((str(vid).lower(), float(item["start_sec"]), float(item["end_sec"]),
                                    item.get("title") or result.get("title") or "Video"))

    def video_ids(self) -> set[str]:
        return {r[0] for r in self.ranges}

    def resolve_video(self, token: str) -> str | None:
        """The evidence video a cited id refers to: exact, or a unique prefix of at least MIN_PREFIX characters."""
        t = token.lower()
        ids = self.video_ids()
        if t in ids:
            return t
        if len(t) >= MIN_PREFIX:
            matches = [v for v in ids if v.startswith(t)]
            if len(matches) == 1:
                return matches[0]
        return None

    def find(self, video_id: str, seconds: float) -> tuple[str, float, float, str] | None:
        for r in self.ranges:
            if r[0] == video_id and r[1] - TOLERANCE_SEC <= seconds <= r[2] + TOLERANCE_SEC:
                return r
        return None


def validate_citations(text: str, evidence: Evidence) -> tuple[str, list[Citation]]:
    """Replaces valid [video_id@seconds] markers by [n], drops invalid ones, returns (text, citations in order of first use)."""
    numbers: dict[tuple[str, int], int] = {}
    cited: dict[int, Citation] = {}
    dropped = False

    def replace(m: re.Match) -> str:
        nonlocal dropped
        out: list[int] = []
        for ref in _ONE.finditer(m.group(0)):
            vid = evidence.resolve_video(ref.group(1))
            seconds = float(ref.group(2))
            hit = evidence.find(vid, seconds) if vid else None
            if hit is None:
                dropped = True
                continue
            key = (hit[0], int(seconds))
            n = numbers.setdefault(key, len(numbers) + 1)
            cited.setdefault(n, Citation(n=n, video_id=hit[0], title=hit[3], start_sec=float(int(seconds)), end_sec=hit[2]))
            if n not in out:
                out.append(n)
        return "".join(f"[{n}]" for n in out)

    cleaned = _GROUP.sub(replace, text)
    if dropped:  # tidy the gap a removed marker leaves ("word [x@9]." -> "word.")
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), [cited[n] for n in sorted(cited)]


def fallback_answer(evidence: Evidence, limit: int = 3) -> str:
    """When the model could not finish (call cap, daily budget): say so and point at the best moments it did find."""
    if not evidence.ranges:
        return NO_CONTEXT_ANSWER
    lines, seen = [], set()
    for vid, start, _end, title in evidence.ranges:
        key = (vid, int(start))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{vid}@{int(start)}] {title} at {int(start) // 60:02d}:{int(start) % 60:02d}")
        if len(lines) == limit:
            break
    return f"{UNFINISHED_PREFIX}, but these are the most relevant moments I found:\n" + "\n".join(lines)
