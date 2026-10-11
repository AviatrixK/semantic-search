"""Duplicate upload detection: the SHA-256 of the uploaded file is stored in videos.sha256 (unique)."""
import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Video

CHUNK = 1024 * 1024


def sha256_of_fileobj(fileobj) -> str:
    """Hex SHA-256 of a seekable binary file, streamed in 1 MiB pieces (memory stays flat for 500 MB uploads). The file is
    left positioned at the start, ready to be uploaded."""
    fileobj.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: fileobj.read(CHUNK), b""):
        digest.update(block)
    fileobj.seek(0)
    return digest.hexdigest()


def find_by_sha256(db: Session, digest: str) -> Video | None:
    return db.scalars(select(Video).where(Video.sha256 == digest).limit(1)).first()


def duplicate_message(existing: Video) -> str:
    """What the admin sees. Says which video it is and what to do next, depending on its state."""
    base = f'This exact file was already uploaded as "{existing.title}" (video {existing.id}).'
    if existing.status == "failed":
        return base + " Processing it failed: use Reprocess on that video instead of uploading it again."
    if existing.status != "ready":
        return base + " It is still being processed."
    return base + " Delete that video first if you want to upload it again."


def duplicate_body(existing: Video) -> dict:
    return {"detail": duplicate_message(existing), "video_id": str(existing.id), "title": existing.title,
            "status": existing.status}
