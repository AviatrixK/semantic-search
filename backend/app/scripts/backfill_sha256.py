"""Fills videos.sha256 for videos uploaded before duplicate detection existed.

It streams each stored object through SHA-256 (nothing is held in memory or written to disk). `--dry-run` computes and
reports everything but writes nothing; it still has to read every object once, so it costs the same transfer as the real run.

Run it BEFORE remuxing old videos to faststart (`remux_faststart`): remuxing rewrites the stored bytes, so a hash taken
afterwards is not the hash of the file the admin originally selected and would not catch a re-upload of it. When a backup
`originals/<key>` exists (remux_faststart --keep-original) that copy is hashed instead, for the same reason.

Duplicates already in the library are reported, not deleted and not hashed (the unique index forbids two videos with one
hash): the first video keeps the hash, the later ones stay NULL, and you decide what to delete.

Usage (inside the api container):
  python -m app.scripts.backfill_sha256 --dry-run
  python -m app.scripts.backfill_sha256
  python -m app.scripts.backfill_sha256 --video-id <uuid>
Exit code 1 if any video could not be read. Safe to re-run: videos that already have a hash are skipped.
"""
import argparse
import sys
import uuid

from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import Video
from app.services import storage


def source_key(video_key: str) -> tuple[str, str]:
    """(key to hash, note). Prefers the untouched original kept by `remux_faststart --keep-original`."""
    backup = f"originals/{video_key}"
    if storage.object_exists(backup):
        return backup, "from originals/ backup"
    return video_key, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--video-id", type=uuid.UUID)
    args = ap.parse_args(argv)

    counts: dict[str, int] = {}
    failed = 0
    with SessionLocal() as db:
        owner: dict[str, tuple[uuid.UUID, str]] = {  # hash -> (video id, title) for hashes already stored
            h: (i, t) for i, t, h in db.execute(select(Video.id, Video.title, Video.sha256).where(Video.sha256.is_not(None)))}
        stmt = select(Video).where(Video.sha256.is_(None)).order_by(Video.created_at)
        if args.video_id:
            stmt = stmt.where(Video.id == args.video_id)
        videos = db.scalars(stmt).all()
        if not videos:
            print("Nothing to do: every video already has a hash.")
            return 0
        for video in videos:
            try:
                key, note = source_key(video.storage_key)
                digest = storage.sha256_of_object(key)
            except Exception as exc:
                failed += 1
                result, note = "FAILED", f"{type(exc).__name__}: {str(exc)[:120]}"
            else:
                if digest in owner:
                    other_id, other_title = owner[digest]
                    result, note = "DUPLICATE", f'same file as {other_id} "{other_title}"; left empty'
                else:
                    owner[digest] = (video.id, video.title)
                    result = "would-set" if args.dry_run else "set"
                    if not args.dry_run:
                        video.sha256 = digest
                        db.commit()
            counts[result] = counts.get(result, 0) + 1
            print(f"{video.id}  {video.title[:36]!s:36}  {result:9} {note}")
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + ("  (dry run: nothing written)" if args.dry_run else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
