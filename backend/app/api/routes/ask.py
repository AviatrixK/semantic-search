import json
import logging
import queue
import threading
from typing import Any, Callable, Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent import orchestrator
from app.agent.limits import Cancelled
from app.api.deps import CurrentUser, current_user
from app.core.config import settings
from app.core.db import SessionLocal, get_db
from app.core.ratelimit import ask_rate_limit
from app.schemas.ask import SESSION_ID_PATTERN, AskIn, AskOut, CitationOut, TraceStepOut, UsageOut
from app.services import retrieval
from app.services.llm import LLMError
from app.services.memory import ChatMemory
from app.services.usage import DailyBudgetExceeded, UsageMeter

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ask"])

GENERIC_ERROR = "Something went wrong while answering. Please try again."


# Dependencies (so tests can substitute fakes without touching global state)
def get_session_factory() -> Callable[[], Any]:
    return SessionLocal


def get_meter() -> UsageMeter:
    return UsageMeter()


def get_memory() -> ChatMemory:
    return ChatMemory()


def to_out(result: orchestrator.AskResult) -> AskOut:
    return AskOut(
        answer=result.answer, mode=result.requested, route=result.route, route_reason=result.route_reason,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        trace=[TraceStepOut(**s.__dict__) for s in result.trace],
        usage=UsageOut(llm_calls=result.llm_calls, tokens=result.tokens),
    )


def _budget_http_error(exc: DailyBudgetExceeded) -> HTTPException:
    return HTTPException(429, exc.user_message, headers={"Retry-After": str(exc.retry_after)})


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn, user: CurrentUser = Depends(current_user), db: Session = Depends(get_db),
        factory: Callable[[], Any] = Depends(get_session_factory), meter: UsageMeter = Depends(get_meter),
        memory: ChatMemory = Depends(get_memory), _: None = Depends(ask_rate_limit)):
    retrieval.log_query(db, user.id, body.question, "ask")
    try:
        result = orchestrator.ask(factory, user.id, body.question, mode=body.mode, session_id=body.session_id,
                                  meter=meter, memory=memory)
    except DailyBudgetExceeded as exc:
        raise _budget_http_error(exc)
    except LLMError as exc:
        log.warning("ask failed for user %s: %s: %s", user.id, type(exc).__name__, exc)
        raise HTTPException(503, exc.user_message)  # short and safe: never the SDK's message
    return to_out(result)


# ---- Server-Sent Events: progress while the agent works, then the answer
def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def event_stream(factory, user_id: str, question: str, mode: str, session_id: str | None, meter: UsageMeter,
                 memory: ChatMemory, keepalive_sec: float) -> Iterator[str]:
    """Runs the request in a worker thread and relays its events. If the client disconnects, this generator is closed,
    `cancel` is set, and the agent stops before its next model call instead of burning tokens for nobody."""
    events: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def work() -> None:
        try:
            result = orchestrator.ask(factory, user_id, question, mode=mode, session_id=session_id, meter=meter, memory=memory,
                                      on_event=lambda e: events.put((e["type"], e)), should_stop=cancel.is_set)
            events.put(("answer", to_out(result).model_dump()))
        except Cancelled:
            pass
        except DailyBudgetExceeded as exc:
            events.put(("error", {"message": exc.user_message, "status": 429, "retry_after": exc.retry_after}))
        except LLMError as exc:
            log.warning("ask stream failed for user %s: %s: %s", user_id, type(exc).__name__, exc)
            events.put(("error", {"message": exc.user_message, "status": 503}))
        except Exception:
            log.exception("ask stream crashed")
            events.put(("error", {"message": GENERIC_ERROR, "status": 500}))
        finally:
            events.put(("done", {}))

    threading.Thread(target=work, daemon=True, name="ask-stream").start()
    try:
        while True:
            try:
                kind, payload = events.get(timeout=keepalive_sec)
            except queue.Empty:
                yield ": keep-alive\n\n"  # a comment line: ignored by clients, keeps proxies from closing an idle stream
                continue
            yield sse(kind, payload)
            if kind == "done":
                return
    finally:
        cancel.set()


@router.get("/ask/stream")
def ask_stream(
    q: str = Query(min_length=3, max_length=1000), mode: Literal["rag", "agent", "auto"] = "auto",
    session_id: str | None = Query(None, pattern=SESSION_ID_PATTERN),
    user: CurrentUser = Depends(current_user), factory: Callable[[], Any] = Depends(get_session_factory),
    meter: UsageMeter = Depends(get_meter), memory: ChatMemory = Depends(get_memory), _: None = Depends(ask_rate_limit),
):
    question = " ".join(q.split())
    if len(question) < 3:
        raise HTTPException(422, "Question is too short")
    try:
        meter.check(user.id)  # a real HTTP 429 before the stream starts, so clients can show Retry-After normally
    except DailyBudgetExceeded as exc:
        raise _budget_http_error(exc)
    with factory() as db:
        retrieval.log_query(db, user.id, question, "ask")
    return StreamingResponse(
        event_stream(factory, user.id, question, mode, session_id, meter, memory, settings.SSE_KEEPALIVE_SEC),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},  # no buffering by proxies
    )


@router.delete("/ask/session/{session_id}", status_code=204)
def forget_session(session_id: str = Path(pattern=SESSION_ID_PATTERN), user: CurrentUser = Depends(current_user),
                   memory: ChatMemory = Depends(get_memory)):
    memory.clear(user.id, session_id)
