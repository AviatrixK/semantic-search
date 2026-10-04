"""Add gold entries to eval/queries.jsonl while you watch a video.

    docker compose exec api python eval/add_query.py                    # pick a video from a list
    docker compose exec api python eval/add_query.py --video "purpose"  # a title fragment or the video UUID

For each entry: type the question you would ask, say whether it is an exact / paraphrase / multi query, then the moment that
answers it (2:03-2:40, 123-160, 1m30s-2m). The transcript of that moment is shown so you can check you labelled the right spot.
At the question prompt: blank = finish, v = change video, u = undo the last entry you saved, l = show this video's list again.
Giving the SAME question text again adds another correct moment to it (useful for `multi` questions)."""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app` and `eval` import when run as a script

from eval import metrics  # noqa: E402

TYPE_KEYS = {"e": "exact", "p": "paraphrase", "m": "multi", **{t: t for t in metrics.QUERY_TYPES}}
PREVIEW_CHARS = 300


def transcript_preview(db, video_id: str, start: float, end: float) -> tuple[str | None, str | None]:
    """(text of the chunks overlapping the range or None, a warning or None)."""
    from app.models import Video
    from app.services import windowing

    window = windowing.get_window(db, uuid.UUID(video_id), start, end)
    if window is None:
        return None, "this video is not in the database"
    video = db.get(Video, uuid.UUID(video_id))
    warn = None
    if video is not None and video.duration_sec and end > video.duration_sec + 1:
        warn = f"the end is past the video's length ({metrics.clock(video.duration_sec)})"
    text = " ".join(s["text"] for s in window["segments"]).strip()
    if not text:
        return None, warn or "no transcript text overlaps that range"
    return (text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS].rstrip() + "..."), warn


def choose_video(db, ask, say, hint: str | None) -> dict | None:
    from app.services import catalog

    videos = catalog.list_ready_videos(db, title_contains=None if not hint or _is_uuid(hint) else hint, limit=50)
    if hint and _is_uuid(hint):
        videos = [v for v in catalog.list_ready_videos(db, limit=200) if v["video_id"] == str(uuid.UUID(hint))]
    if len(videos) == 1:
        say(f"Video: {videos[0]['title']}")
        return videos[0]
    if not videos:
        say("No ready videos match." if hint else "There are no ready videos yet.")
        return None
    while True:
        for i, v in enumerate(videos, start=1):
            dur = metrics.clock(v["duration_sec"]) if v["duration_sec"] else "?"
            say(f"  {i}. {v['title']}  ({dur})  {v['video_id']}")
        choice = ask("Video number (blank to quit): ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            return videos[int(choice) - 1]
        say("Not a number in the list.")


def _is_uuid(text: str) -> bool:
    try:
        uuid.UUID(text)
        return True
    except ValueError:
        return False


def count_line(path: Path) -> str:
    if not path.exists():
        return "0 entries"
    queries = metrics.load_queries(path)
    per_type = ", ".join(f"{sum(1 for q in queries if q.type == t)} {t}" for t in metrics.QUERY_TYPES)
    return f"{sum(len(q.gold) for q in queries)} entries, {len(queries)} queries ({per_type}), {len({g.video_id for q in queries for g in q.gold})} videos"


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(("\n" if needs_newline else "") + line + "\n")


def undo_last(path: Path, session_lines: list[str]) -> str | None:
    """Removes the most recent line saved in THIS session (never older entries). Returns it, or None."""
    if not session_lines or not path.exists():
        return None
    line = session_lines.pop()
    lines = path.read_text(encoding="utf-8").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == line:
            del lines[i]
            path.write_text("".join(f"{x}\n" for x in lines), encoding="utf-8", newline="\n")
            return line
    return None


def run(db, path: Path, *, ask=input, say=print, video_hint: str | None = None) -> int:
    """The interactive loop. Returns the number of entries saved. ask/say are injectable so tests can script a session."""
    saved: list[str] = []
    video = choose_video(db, ask, say, video_hint)
    if video is None:
        return 0
    say(f"\nLabelling '{video['title']}'. Currently: {count_line(path)}.")
    try:
        while True:
            query = ask("\nQuestion (blank=done, v=change video, u=undo, l=list): ").strip()
            if not query:
                break
            if query.lower() == "v":
                video = choose_video(db, ask, say, None) or video
                say(f"Now labelling '{video['title']}'.")
                continue
            if query.lower() == "l":
                continue
            if query.lower() == "u":
                gone = undo_last(path, saved)
                say(f"Removed: {gone}" if gone else "Nothing from this session to undo.")
                continue
            qtype = TYPE_KEYS.get(ask("Type: [e]xact words / [p]araphrase / [m]ulti-part: ").strip().lower())
            if qtype is None:
                say("Please answer e, p or m. Entry skipped.")
                continue
            try:
                start, end = metrics.parse_range(ask("Moment that answers it (e.g. 2:03-2:40): "))
            except ValueError as exc:
                say(f"{exc}. Entry skipped.")
                continue
            text, warn = transcript_preview(db, video["video_id"], start, end)
            say(f"  {metrics.clock(start)}-{metrics.clock(end)}: {text or '(no transcript text)'}")
            if warn:
                say(f"  WARNING: {warn}")
            if ask("Save? [Y/n]: ").strip().lower() in ("n", "no"):
                say("Skipped.")
                continue
            line = metrics.entry_json(query, qtype, video["video_id"], start, end)
            if path.exists() and line in path.read_text(encoding="utf-8").splitlines():
                say("That exact entry is already in the file.")
                continue
            append_line(path, line)
            saved.append(line)
            say(f"Saved. Now: {count_line(path)}.")
    except (EOFError, KeyboardInterrupt):
        say("\nStopped.")
    return len(saved)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", type=Path, default=metrics.DEFAULT_QUERIES)
    ap.add_argument("--video", help="video UUID or part of its title")
    args = ap.parse_args(argv)
    from app.core.db import SessionLocal
    with SessionLocal() as db:
        n = run(db, args.file, video_hint=args.video)
    print(f"{n} entr{'y' if n == 1 else 'ies'} added to {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
