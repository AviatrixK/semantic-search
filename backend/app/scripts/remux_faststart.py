"""Rewrites already-stored MP4/MOV videos so their index ("moov") is at the start of the file, replacing the object in S3
in place. No re-encode (ffmpeg stream copy): seconds per video, no quality loss. Videos that are already faststart, and
webm/mkv (they have no moov), are skipped, so it is safe to run again.

Each file is downloaded, remuxed, verified (index first, duration within 1 s, same number of audio streams) and only then
uploaded over the original. A failure leaves that video untouched and the script goes on with the next.

Usage (inside the api container):
  python -m app.scripts.remux_faststart --dry-run                 # list what would change, touch nothing
  python -m app.scripts.remux_faststart                           # fix every video that needs it
  python -m app.scripts.remux_faststart --video-id <uuid>         # one video
  python -m app.scripts.remux_faststart --keep-original           # also copy each original to originals/<key> first
Exit code 1 if any video failed.
"""
import argparse
import os
import sys
import tempfile
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Job, Video
from app.services import jobevents, media, storage


def process(key: str, tmp: str, *, dry_run: bool, keep_original: bool) -> str:
    """Returns "fixed", "would-fix" or "already-fast"/"skipped:<why>". Raises on a problem (the caller reports it)."""
    ext = key.rsplit(".", 1)[-1].lower()
    if ext not in media.FASTSTART_TYPES:
        return f"skipped: .{ext} has no moov atom"
    if media.moov_before_mdat_in_head(storage.read_head(key)) is True:
        return "already-fast"  # decided from the first 64 KB: nothing downloaded
    src = os.path.join(tmp, f"src.{ext}")
    storage.download_to(key, src)
    state = media.moov_before_mdat(src)
    if state is True:
        return "already-fast"
    if state is None:
        return "skipped: not an MP4/MOV container"
    if dry_run:
        return "would-fix"
    out = media.remux_faststart(src, os.path.join(tmp, f"fast.{ext}"))
    if abs(media.duration_sec(out) - media.duration_sec(src)) > 1:
        raise media.MediaError("remuxed duration differs from the original; not replacing it")
    if media.has_audio(out) != media.has_audio(src):
        raise media.MediaError("remuxed file lost its audio; not replacing it")
    if keep_original:
        with open(src, "rb") as f:
            storage.upload_fileobj(f, f"originals/{key}", media.FASTSTART_TYPES[ext])
    with open(out, "rb") as f:
        storage.upload_fileobj(f, key, media.FASTSTART_TYPES[ext])
    return "fixed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--video-id", type=uuid.UUID)
    ap.add_argument("--keep-original", action="store_true")
    args = ap.parse_args(argv)

    with SessionLocal() as db:
        stmt = select(Video.id, Video.title, Video.storage_key).order_by(Video.created_at)
        if args.video_id:
            stmt = stmt.where(Video.id == args.video_id)
        videos = db.execute(stmt).all()
        busy = set(db.scalars(select(Job.video_id).where(Job.stage.in_(jobevents.ACTIVE_STAGES))).all())
    if not videos:
        print("No videos found.")
        return 0

    counts: dict[str, int] = {}
    failed = 0
    for video_id, title, key in videos:
        try:
            if video_id in busy:  # the worker may be replacing this very object
                result = "skipped: a processing job is running for it"
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    result = process(key, tmp, dry_run=args.dry_run, keep_original=args.keep_original)
        except Exception as exc:  # one broken file must not stop the rest
            failed += 1
            result = f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"
        kind = result.split(":")[0]
        counts[kind] = counts.get(kind, 0) + 1
        print(f"{video_id}  {title[:40]!s:40}  {result}")
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  (bucket {settings.S3_BUCKET})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
