import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user
from app.core.db import get_db
from app.core.ratelimit import search_rate_limit
from app.schemas.video import SearchHit
from app.services import retrieval

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=list[SearchHit])
def search(q: str = Query(min_length=2, max_length=500), k: int = Query(10, ge=1, le=50),
           video_id: uuid.UUID | None = None,
           user: CurrentUser = Depends(current_user), db: Session = Depends(get_db),
           _: None = Depends(search_rate_limit)):
    retrieval.log_query(db, user.id, q, "search")
    return retrieval.vector_search(db, q, k=k, video_id=video_id)
