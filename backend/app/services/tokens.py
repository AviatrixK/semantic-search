from sqlalchemy import delete, func, update
from sqlalchemy.orm import Session

from app.models import RefreshToken


def revoke_all_for_user(db: Session, user_id) -> int:
    """Revoke every still-valid refresh token of a user (used when a revoked token is replayed)."""
    res = db.execute(update(RefreshToken).where(RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False))
                     .values(revoked=True))
    return res.rowcount


def purge_expired_refresh_tokens(db: Session) -> int:
    """Delete expired rows. Revoked-but-unexpired rows are kept: they are how token reuse is detected."""
    res = db.execute(delete(RefreshToken).where(RefreshToken.expires_at < func.now()))
    db.commit()
    return res.rowcount
