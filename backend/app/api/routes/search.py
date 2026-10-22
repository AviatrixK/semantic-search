import logging
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
from app.services.timing import Timings

log = logging.getLogger(__name__)

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
    timings = Timings()
    with timings.measure("log"):
        retrieval.log_query(db, user.id, q, "search")
    run = {"vector": retrieval.vector_search, "keyword": retrieval.keyword_search, "hybrid": retrieval.hybrid_search}[mode]
    hits = run(db, q, k=k, video_id=video_id, uploaded_after=uploaded_after, highlight=highlight, timings=timings)
    total_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Search-Ms"] = f"{total_ms:.1f}"
    response.headers["Server-Timing"] = timings.server_timing(total_ms)  # DevTools > Network > Timing shows each part
    response.headers["X-Search-Highlight"] = timings.highlight_summary()  # stored = from chunk_sentences, fallback = embedded now
    if timings.counts.get("embed_failed"):  # the embedding provider was down or too slow: this answer is keyword-only
        response.headers["X-Search-Fallback"] = "keyword-only"
    log.debug("search mode=%s k=%d hits=%d %s highlight[%s]", mode, k, len(hits), timings.server_timing(total_ms),
              timings.highlight_summary())
    return hits
