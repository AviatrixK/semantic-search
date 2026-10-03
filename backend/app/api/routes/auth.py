from datetime import datetime, timedelta, timezone

import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user
from app.core.config import settings
from app.core.db import get_db
from app.core.ratelimit import login_rate_limit
from app.core.security import (create_access_token, hash_password, hash_refresh_token,
                               new_refresh_token, verify_password)
from app.models import RefreshToken, User
from app.schemas.auth import LoginIn, RegisterIn, TokenOut, UserOut
from app.services import tokens

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
COOKIE = "refresh_token"


def _issue_tokens(db: Session, user: User, response: Response) -> TokenOut:
    raw, hashed = new_refresh_token()
    db.add(RefreshToken(user_id=user.id, token_hash=hashed,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_DAYS)))
    db.commit()
    response.set_cookie(COOKIE, raw, httponly=True, secure=settings.COOKIE_SECURE, samesite="lax",
                        max_age=settings.REFRESH_TOKEN_DAYS * 86400, path="/auth")
    return TokenOut(access_token=create_access_token(str(user.id), user.role))


@router.post("/register", response_model=UserOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    user = User(email=email, password_hash=hash_password(body.password), role="user")  # never admin here
    db.add(user)
    try:
        db.commit()
    except IntegrityError:  # lost a race with a concurrent registration of the same email
        db.rollback()
        raise HTTPException(409, "Email already registered")
    return UserOut(id=str(user.id), email=user.email, role=user.role)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, response: Response, db: Session = Depends(get_db), _: None = Depends(login_rate_limit)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")  # generic on purpose
    return _issue_tokens(db, user, response)


@router.post("/refresh", response_model=TokenOut)
def refresh(response: Response, refresh_token: str | None = Cookie(None), db: Session = Depends(get_db)):
    if not refresh_token:
        raise HTTPException(401, "No refresh token")
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
    if not row:
        raise HTTPException(401, "Refresh token invalid")
    if row.revoked:
        # A rotated or logged-out token is being replayed: assume it leaked and end every session of this user.
        _revoke_everything(db, row, "revoked refresh token presented")
    if row.expires_at < datetime.now(timezone.utc):
        raise HTTPException(401, "Refresh token invalid")
    # Atomic single-use claim: only one of two concurrent requests with the same token can flip revoked.
    claimed = db.execute(update(RefreshToken).where(RefreshToken.id == row.id, RefreshToken.revoked.is_(False))
                         .values(revoked=True))
    if claimed.rowcount != 1:
        _revoke_everything(db, row, "refresh token used concurrently")
    user = db.get(User, row.user_id)
    return _issue_tokens(db, user, response)  # commits the revoke + the new token together


def _revoke_everything(db: Session, row: RefreshToken, reason: str):
    n = tokens.revoke_all_for_user(db, row.user_id)
    db.commit()
    log.warning("auth: %s for user %s; revoked %s active refresh token(s)", reason, row.user_id, n)
    raise HTTPException(401, "Refresh token invalid")


@router.post("/logout", status_code=204)
def logout(response: Response, refresh_token: str | None = Cookie(None), db: Session = Depends(get_db)):
    if refresh_token:
        row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
        if row:
            row.revoked = True
            db.commit()
    response.delete_cookie(COOKIE, path="/auth")


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser = Depends(current_user), db: Session = Depends(get_db)):
    u = db.get(User, user.id)
    if u is None:  # valid token for a user that no longer exists
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return UserOut(id=str(u.id), email=u.email, role=u.role)
