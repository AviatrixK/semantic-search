"""The agent's tools. Each has (1) a JSON schema the model sees, (2) a Pydantic model that validates what the model actually
sent, and (3) an executor that calls app.services only (never SQL here). Bad arguments never raise: the model gets an error
message back and can fix its call. Transcript text leaves here wrapped as untrusted data."""
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.config import settings
from app.services import catalog, retrieval, windowing
from app.services.llm import ToolSpec
from app.services.safety import sanitize_untrusted, wrap_excerpt

log = logging.getLogger(__name__)

LIST_KEYS = ("clips", "videos", "segments")
TRANSCRIPT_OPEN, TRANSCRIPT_CLOSE = "<transcript_excerpt>", "</transcript_excerpt>"


def mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def ref(video_id: str, start_sec: float) -> str:
    """What the model cites: [<video_id>@<seconds>]."""
    return f"{video_id}@{int(start_sec)}"


# ---- argument models (what the model sent, validated)
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo like "video" instead of "video_id" must not silently do nothing


class SearchArgs(_Args):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(5, ge=1, le=10)
    video_id: uuid.UUID | None = None

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("String should have at least 2 characters")
        return v


class ListVideosArgs(_Args):
    uploaded_after: date | None = None
    max_duration_sec: int | None = Field(None, ge=1)
    title_contains: str | None = Field(None, min_length=1, max_length=100)


class WindowArgs(_Args):
    video_id: uuid.UUID
    start_sec: float = Field(ge=0)
    span_sec: float = Field(60, ge=1, le=300)


# ---- executors: (db session, validated args) -> JSON-able dict
def search_transcripts(db, a: SearchArgs) -> dict:
    hits = retrieval.hybrid_search(db, a.query, k=a.k, video_id=a.video_id, highlight=False)
    clips = [{
        "ref": ref(h["video_id"], h["start_sec"]), "video_id": h["video_id"], "title": sanitize_untrusted(h["title"]),
        "start_sec": h["start_sec"], "end_sec": h["end_sec"], "time": f"{mmss(h['start_sec'])}-{mmss(h['end_sec'])}",
        "score": h["score"], "text": wrap_excerpt(h["text"]),
    } for h in hits]
    out: dict[str, Any] = {"query": a.query, "count": len(clips), "clips": clips}
    if not clips:
        out["note"] = "No clips matched. Try different or broader words, or remove the video_id filter."
    return out


def list_videos(db, a: ListVideosArgs) -> dict:
    rows = catalog.list_ready_videos(db, uploaded_after=a.uploaded_after, max_duration_sec=a.max_duration_sec,
                                     title_contains=a.title_contains)
    videos = [{"video_id": v["video_id"], "title": sanitize_untrusted(v["title"]), "duration_sec": v["duration_sec"],
               "duration": mmss(v["duration_sec"]) if v["duration_sec"] is not None else None, "uploaded": v["uploaded"]}
              for v in rows]
    out: dict[str, Any] = {"count": len(videos), "videos": videos}
    if not videos:
        out["note"] = "No videos match these filters."
    return out


def get_transcript_window(db, a: WindowArgs) -> dict:
    end = a.start_sec + a.span_sec
    win = windowing.get_window(db, a.video_id, a.start_sec, end)
    if win is None:
        return {"error": f"No video with id {a.video_id}. Use a video_id from search_transcripts or list_videos."}
    segments = [{"ref": ref(win["video_id"], s["start_sec"]), "start_sec": s["start_sec"], "end_sec": s["end_sec"],
                 "time": f"{mmss(s['start_sec'])}-{mmss(s['end_sec'])}", "text": wrap_excerpt(s["text"])}
                for s in win["segments"]]
    out: dict[str, Any] = {"video_id": win["video_id"], "title": sanitize_untrusted(win["title"]), "start_sec": a.start_sec,
                           "end_sec": end, "segments": segments}
    if not segments:
        out["note"] = "No transcript text in that range."
    return out


# ---- registry (schemas are written by hand: simple types the model API accepts, no anyOf/null unions)
@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    model: type[BaseModel]
    run: Callable[[Any, Any], dict]


TOOLS: dict[str, Tool] = {t.spec.name: t for t in (
    Tool(ToolSpec(
        name="search_transcripts",
        description="Search the transcripts of all videos by meaning AND exact words (names, acronyms). Returns the best-matching clips with a `ref` to cite, "
                    "the video title, the time range and the transcript text. Search once per sub-question.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in natural language (2-500 characters)."},
            "k": {"type": "integer", "description": "How many clips to return, 1-10 (default 5)."},
            "video_id": {"type": "string", "description": "Optional: only search inside this video (a UUID from earlier results)."},
        }, "required": ["query"]}), SearchArgs, search_transcripts),
    Tool(ToolSpec(
        name="list_videos",
        description="List the videos in the library (ready ones), newest first, with title, duration and upload date. "
                    "Use it for questions about which videos exist, or to filter by upload date, length or title.",
        parameters={"type": "object", "properties": {
            "uploaded_after": {"type": "string", "description": "Optional: only videos uploaded on or after this date, YYYY-MM-DD."},
            "max_duration_sec": {"type": "integer", "description": "Optional: only videos at most this many seconds long."},
            "title_contains": {"type": "string", "description": "Optional: only videos whose title contains this text."},
        }}), ListVideosArgs, list_videos),
    Tool(ToolSpec(
        name="get_transcript_window",
        description="Read the transcript of one video from start_sec for span_sec seconds, to get context around a clip. "
                    "Each segment has a `ref` to cite.",
        parameters={"type": "object", "properties": {
            "video_id": {"type": "string", "description": "The video's UUID, copied from earlier results."},
            "start_sec": {"type": "number", "description": "Where to start reading, in seconds (0 or more)."},
            "span_sec": {"type": "number", "description": "How many seconds to read, 1-300 (default 60)."},
        }, "required": ["video_id", "start_sec"]}), WindowArgs, get_transcript_window),
)}
TOOL_SPECS: list[ToolSpec] = [t.spec for t in TOOLS.values()]


# ---- running a call safely
@dataclass(frozen=True)
class ToolOutcome:
    result: dict
    ok: bool


def format_validation_error(e: ValidationError, allowed: list[str]) -> str:
    problems = []
    for err in e.errors()[:5]:
        field = ".".join(str(x) for x in err["loc"]) or "arguments"
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = f"not an allowed argument (allowed: {', '.join(allowed)})"
        problems.append(f"{field}: {msg}")
    return "Invalid arguments: " + "; ".join(problems) + ". Fix the arguments and call the tool again."


def execute_tool(session_factory: Callable[[], Any], name: str, args: Any) -> ToolOutcome:
    """Validate, run, and ALWAYS return something the model can read: errors come back as {"error": "..."}."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolOutcome({"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."}, False)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ToolOutcome({"error": "Invalid arguments: expected a JSON object. Fix the arguments and call the tool again."}, False)
    try:
        parsed = tool.model.model_validate(args)
    except ValidationError as e:
        return ToolOutcome({"error": format_validation_error(e, list(tool.model.model_fields))}, False)
    try:
        with session_factory() as db:  # one short session per call: tools may run in parallel threads
            result = tool.run(db, parsed)
    except Exception:
        log.exception("agent tool %s failed", name)
        return ToolOutcome({"error": "The tool failed unexpectedly. Try a different approach, or answer with what you have."}, False)
    return ToolOutcome(result, "error" not in result)


# ---- token budget for what goes back to the model
def estimate_tokens(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False)) // 4 + 1  # about 4 characters per token


def _clip_text(text: str, max_chars: int) -> str:
    """Shortens an excerpt but keeps it a well-formed <transcript_excerpt> block."""
    if text.startswith(TRANSCRIPT_OPEN) and text.endswith(TRANSCRIPT_CLOSE):
        inner = text[len(TRANSCRIPT_OPEN):-len(TRANSCRIPT_CLOSE)]
        return f"{TRANSCRIPT_OPEN}{inner[:max_chars].rstrip()}…{TRANSCRIPT_CLOSE}" if len(inner) > max_chars else text
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def truncate_result(result: dict, max_tokens: int | None = None) -> dict:
    """Keeps a tool result within a token budget: drops trailing list items (the least relevant ones come last), then
    shortens the text of the last one if it alone is still too big. Says so in the result so the model knows it is partial."""
    budget = settings.AGENT_TOOL_RESULT_TOKENS if max_tokens is None else max_tokens
    if estimate_tokens(result) <= budget:
        return result
    key = next((k for k in LIST_KEYS if isinstance(result.get(k), list) and result[k]), None)
    if key is None:  # nothing to drop item by item: cut the raw JSON
        return {"truncated": True, "note": "Result was too long and was cut.",
                "content": json.dumps(result, ensure_ascii=False)[: budget * 4]}
    total = len(result[key])
    items = list(result[key])

    def build(its: list) -> dict:
        out = {**result, key: its, "truncated": True, "omitted": total - len(its)}
        out["note"] = (result.get("note", "") + f" Showing {len(its)} of {total}. Ask for fewer items or a narrower query or window.").strip()
        return out

    while len(items) > 1 and estimate_tokens(build(items)) > budget:
        items.pop()
    limit = max(len(i.get("text", "")) for i in items) or 0
    while estimate_tokens(build(items)) > budget and limit > 120 and any("text" in i for i in items):
        limit = int(limit * 0.8)
        items = [{**i, "text": _clip_text(i["text"], limit)} if "text" in i else i for i in items]
    return build(items)


# ---- one-line descriptions for the trace the user sees
def describe_call(name: str, args: Any) -> str:
    a = args if isinstance(args, dict) else {}
    if name == "search_transcripts":
        q = " ".join(str(a.get("query", "")).split())
        return f"Searching “{q[:80]}”" if q else "Searching"
    if name == "list_videos":
        bits = []
        if a.get("title_contains"):
            bits.append(f"title contains “{str(a['title_contains'])[:40]}”")
        if a.get("uploaded_after"):
            bits.append(f"uploaded after {a['uploaded_after']}")
        if a.get("max_duration_sec"):
            bits.append(f"up to {mmss(float(a['max_duration_sec']))} long")
        return "Listing videos" + (f" ({', '.join(bits)})" if bits else "")
    if name == "get_transcript_window":
        try:
            s = float(a.get("start_sec", 0))
            e = s + float(a.get("span_sec", 60))
            return f"Reading {mmss(s)}–{mmss(e)} of a video"
        except (TypeError, ValueError):
            return "Reading a transcript window"
    return f"Calling {name}"


def summarize_outcome(name: str, args: Any, outcome: ToolOutcome) -> str:
    r = outcome.result
    if not outcome.ok:
        return "Error: " + str(r.get("error", "failed"))[:140]
    if name == "search_transcripts":
        n = r.get("count", 0)
        return f"Found {n} clip{'s' if n != 1 else ''}" if n else "No clips found"
    if name == "list_videos":
        n = r.get("count", 0)
        return f"Found {n} video{'s' if n != 1 else ''}" if n else "No videos found"
    if name == "get_transcript_window":
        return f"Read {mmss(r.get('start_sec', 0))}–{mmss(r.get('end_sec', 0))} of “{r.get('title', 'video')}”"
    return "Done"
