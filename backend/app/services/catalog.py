from datetime import date, datetime, time, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Video


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_ready_videos(db: Session, *, uploaded_after: date | None = None, max_duration_sec: int | None = None,
                      title_contains: str | None = None, limit: int = 20) -> list[dict]:
    """Searchable (ready) videos, newest first. `uploaded_after` is inclusive, from 00:00 UTC of that date."""
    stmt = select(Video).where(Video.status == "ready").order_by(Video.created_at.desc()).limit(limit)
    if uploaded_after:
        stmt = stmt.where(Video.created_at >= datetime.combine(uploaded_after, time.min, tzinfo=timezone.utc))
    if max_duration_sec is not None:
        stmt = stmt.where(Video.duration_sec.is_not(None), Video.duration_sec <= max_duration_sec)
    if title_contains:
        stmt = stmt.where(Video.title.ilike(f"%{_like_escape(title_contains)}%", escape="\\"))
    return [{"video_id": str(v.id), "title": v.title, "duration_sec": v.duration_sec, "uploaded": v.created_at.date().isoformat()}
            for v in db.scalars(stmt)]
