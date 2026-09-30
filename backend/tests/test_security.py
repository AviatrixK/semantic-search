from app.core.security import (create_access_token, decode_access_token, hash_password,
                               hash_refresh_token, new_refresh_token, verify_password)


def test_password_roundtrip():
    h = hash_password("secret123")
    assert verify_password("secret123", h) and not verify_password("wrong", h)


def test_jwt_roundtrip():
    data = decode_access_token(create_access_token("abc", "admin"))
    assert data["sub"] == "abc" and data["role"] == "admin"


def test_refresh_hash():
    raw, h = new_refresh_token()
    assert hash_refresh_token(raw) == h and raw != h
