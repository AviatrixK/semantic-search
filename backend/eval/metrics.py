"""Pure helpers for the evaluation scripts: gold-set loading, time parsing, overlap, rank metrics, citation precision and judge-reply
parsing. No database, no models, no network: everything here is unit-tested in tests/test_eval_metrics.py."""
import json
import re
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path

QUERY_TYPES = ("exact", "paraphrase", "multi")
DEFAULT_QUERIES = Path(__file__).resolve().parent / "queries.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Gold:
    video_id: str
    start_sec: float
    end_sec: float


@dataclass
class Query:
    query: str
    type: str
    gold: list[Gold] = field(default_factory=list)  # one range per labelled entry; a repeated query text adds more ranges


# ---------------------------------------------------------------- time input
_CLOCK = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)$")
_UNITS = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?$")


def parse_time(text: str) -> float:
    """Seconds from 123, 12.5, 2:03, 1:02:03, 1m30s, 90s or 1h2m3s. Raises ValueError for anything else."""
    s = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    m = _CLOCK.match(s)
    if m:
        hours, minutes, seconds = int(m.group(1) or 0), int(m.group(2)), float(m.group(3))
        if seconds >= 60 or (m.group(1) and minutes >= 60):
            raise ValueError(f"not a valid time: {text!r}")
        return hours * 3600 + minutes * 60 + seconds
    m = _UNITS.match(s)
    if m and s:
        h, mi, sec = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0)
    raise ValueError(f"not a valid time: {text!r} (use 123, 2:03, 1:02:03 or 1m30s)")


def parse_range(text: str) -> tuple[float, float]:
    """'2:03-2:40', '2:03 - 2:40', '2:03 to 2:40' or '123 160' -> (start, end). End must be after start."""
    s = text.strip()
    parts = re.split(r"\s*[-–]\s*", s, maxsplit=1) if re.search(r"[-–]", s) else re.split(r"\s+to\s+|\s+", s, maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("give a start and an end, like 2:03-2:40")
    start, end = parse_time(parts[0]), parse_time(parts[1])
    if end <= start:
        raise ValueError("the end must be after the start")
    return start, end


def clock(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------- gold file
def validate_entry(raw: object) -> tuple[str, str, Gold]:
    """One queries.jsonl object -> (query, type, gold range). Raises ValueError with a readable reason."""
    if not isinstance(raw, dict):
        raise ValueError("each line must be a JSON object")
    query = raw.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    qtype = raw.get("type")
    if qtype not in QUERY_TYPES:
        raise ValueError(f"'type' must be one of {', '.join(QUERY_TYPES)}")
    try:
        video_id = str(uuid.UUID(str(raw.get("video_id"))))
    except ValueError:
        raise ValueError("'video_id' must be a video UUID") from None
    start, end = raw.get("start_sec"), raw.get("end_sec")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError("'start_sec' and 'end_sec' must be numbers")
    if start < 0 or end <= start:
        raise ValueError("need 0 <= start_sec < end_sec")
    return query.strip(), qtype, Gold(video_id, float(start), float(end))


def load_queries(path: Path | str) -> list[Query]:
    """Reads the gold file. Lines with the same query text are ONE query with several correct ranges (a 'multi' question
    often needs two moments). Blank lines are skipped. Any bad line raises ValueError naming the file and line."""
    path = Path(path)
    by_text: dict[str, Query] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            query, qtype, gold = validate_entry(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{n}: not valid JSON ({exc.msg})") from None
        except ValueError as exc:
            raise ValueError(f"{path.name}:{n}: {exc}") from None
        entry = by_text.setdefault(query, Query(query, qtype))
        if entry.type != qtype:
            raise ValueError(f"{path.name}:{n}: the query {query!r} appears with two different types")
        if gold not in entry.gold:
            entry.gold.append(gold)
    return list(by_text.values())


def entry_json(query: str, qtype: str, video_id: str, start_sec: float, end_sec: float) -> str:
    """The exact line written to queries.jsonl (validated first)."""
    q, t, g = validate_entry({"query": query, "video_id": video_id, "start_sec": start_sec, "end_sec": end_sec, "type": qtype})
    return json.dumps({"query": q, "video_id": g.video_id, "start_sec": g.start_sec, "end_sec": g.end_sec, "type": t}, ensure_ascii=False)


# ---------------------------------------------------------------- matching and retrieval metrics
def overlaps(video_id: str, start_sec: float, end_sec: float, gold: Gold) -> bool:
    """Same video and the time ranges share some time. Ranges that merely touch (end == start) do not count."""
    return str(video_id).lower() == gold.video_id.lower() and start_sec < gold.end_sec and gold.start_sec < end_sec


def matches_any(video_id: str, start_sec: float, end_sec: float, golds: list[Gold]) -> bool:
    return any(overlaps(video_id, start_sec, end_sec, g) for g in golds)


def first_hit_rank(hits: list[dict], golds: list[Gold]) -> int | None:
    """1-based rank of the first hit that overlaps any gold range, or None when no hit does."""
    for rank, h in enumerate(hits, start=1):
        if matches_any(h["video_id"], h["start_sec"], h["end_sec"], golds):
            return rank
    return None


def retrieval_summary(ranks: list[int | None], latencies_ms: list[float]) -> dict:
    """Recall@1, Recall@5, MRR (a miss counts 0) and the median latency over the queries."""
    n = len(ranks)
    if n == 0:
        return {"n": 0, "recall@1": None, "recall@5": None, "mrr": None, "median_latency_ms": None}
    return {
        "n": n,
        "recall@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
        "median_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
    }


def citation_precision(citations: list, golds: list[Gold]) -> float | None:
    """Share of the answer's citations whose range overlaps a gold range. None when there are no citations (nothing to judge)."""
    if not citations:
        return None
    return sum(1 for c in citations if matches_any(c.video_id, c.start_sec, c.end_sec, golds)) / len(citations)


def mean(values: list[float | int | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def median(values: list[float | int | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.median(kept) if kept else None


# ---------------------------------------------------------------- judge reply
def parse_judge(text: str) -> dict | None:
    """{'groundedness': 1-5 or None, 'relevance': 1-5, 'reason': str} from the judge's reply, tolerating code fences and
    chatter around the JSON. None when no usable scores are found (the caller counts it as a judge failure)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    def score(value) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return value if isinstance(value, int) and 1 <= value <= 5 else None

    groundedness, relevance = score(data.get("groundedness")), score(data.get("relevance"))
    if groundedness is None or relevance is None:
        return None
    return {"groundedness": groundedness, "relevance": relevance, "reason": str(data.get("reason", ""))[:300]}


# ---------------------------------------------------------------- reporting
def fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "-"
    return {"pct": f"{value * 100:.1f}%", "ratio": f"{value:.3f}", "ms": f"{value:.0f} ms", "num": f"{value:.2f}"}[kind]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    line = lambda cells: "| " + " | ".join(cells) + " |"  # noqa: E731
    return "\n".join([line(headers), line(["---"] + ["---:"] * (len(headers) - 1)), *(line(r) for r in rows)])
