"""Usage: docker compose exec api python -m app.scripts.create_admin admin@example.com 'StrongPass123'"""
import sys

from sqlalchemy import select

from app.core.db import SessionLocal
from app.core.security import hash_password
from app.models import User


def main(email: str, password: str):
    db = SessionLocal()
    user = db.scalar(select(User).where(User.email == email.lower()))
    if user:
        user.role = "admin"
        print(f"Promoted existing user {email} to admin")
    else:
        db.add(User(email=email.lower(), password_hash=hash_password(password), role="admin"))
        print(f"Created admin {email}")
    db.commit()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
