import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user
from app.core.db import get_db
from app.core.ratelimit import ask_rate_limit
from app.schemas.ask import AskIn, AskOut, CitationOut
from app.services import rag, retrieval
from app.services.llm import LLMError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ask"])


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn, user: CurrentUser = Depends(current_user), db: Session = Depends(get_db),
        _: None = Depends(ask_rate_limit)):
    retrieval.log_query(db, user.id, body.question, "ask")
    try:
        result = rag.answer(db, body.question)
    except LLMError as exc:
        log.warning("ask failed for user %s: %s: %s", user.id, type(exc).__name__, exc)
        raise HTTPException(503, exc.user_message)  # short and safe: never the SDK's message
    return AskOut(answer=result.answer, mode="rag", citations=[CitationOut(**c.__dict__) for c in result.citations])
