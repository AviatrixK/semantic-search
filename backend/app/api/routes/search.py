import time
import uuid
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user
from app.core.db import get_db
from app.core.ratelimit import search_rate_limit
from app.schemas.video import SearchHit
from app.services import retrieval

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=list[SearchHit])
def search(response: Response, q: str = Query(min_length=2, max_length=500), k: int = Query(10, ge=1, le=50),
           video_id: uuid.UUID | None = None,
           uploaded_after: date | None = Query(None, description="Only videos uploaded on/after this date (UTC)"),
           highlight: bool = Query(True, description="Include the best-matching sentence of each chunk"),
           mode: Literal["vector", "keyword", "hybrid"] = Query("hybrid", description="vector = meaning, keyword = exact words, hybrid = both fused (RRF)"),
           user: CurrentUser = Depends(current_user), db: Session = Depends(get_db),
           _: None = Depends(search_rate_limit)):
    started = time.perf_counter()
    retrieval.log_query(db, user.id, q, "search")
    run = {"vector": retrieval.vector_search, "keyword": retrieval.keyword_search, "hybrid": retrieval.hybrid_search}[mode]
    hits = run(db, q, k=k, video_id=video_id, uploaded_after=uploaded_after, highlight=highlight)
    response.headers["X-Search-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return hits
