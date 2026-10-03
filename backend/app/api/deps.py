import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token

bearer = HTTPBearer(auto_error=False)  # we raise 401 ourselves (HTTPBearer's default is 403)
UNAUTHENTICATED = {"WWW-Authenticate": "Bearer"}


class CurrentUser(dict):
    @property
    def id(self) -> str:
        return self["id"]

    @property
    def role(self) -> str:
        return self["role"]


def current_user(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> CurrentUser:
    if cred is None:
        raise HTTPException(401, "Not authenticated", headers=UNAUTHENTICATED)
    try:
        data = decode_access_token(cred.credentials)
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token", headers=UNAUTHENTICATED)
    return CurrentUser(id=data["sub"], role=data["role"])


def require_admin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(403, "Admins only")
    return user
