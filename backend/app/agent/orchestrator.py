"""One /api/ask request from start to finish: daily budget check -> load chat memory -> choose a route -> answer -> remember.
Both the plain POST and the streaming GET call ask(), so they behave identically; streaming only adds `on_event`."""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agent import router
from app.agent.limits import CallBudget, MeteredLLM
from app.agent.loop import TraceStep, run_agent
from app.core.config import settings
from app.services import rag, retrieval
from app.services.llm import LLM, Message, ModelMsg, UserMsg, get_llm
from app.services.memory import ChatMemory, Turn
from app.services.rag import NO_CONTEXT_ANSWER, Citation, mmss
from app.services.usage import UsageMeter

log = logging.getLogger(__name__)
SEARCH_RESULTS = 5


@dataclass
class AskResult:
    answer: str
    citations: list[Citation]
    route: str  # what was actually used: "search" | "rag" | "agent"
    requested: str  # what the client asked for: "rag" | "agent" | "auto"
    route_reason: str
    trace: list[TraceStep] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0


def search_answer(db, question: str) -> tuple[str, list[Citation]]:
    """The "search" route: no model, just the best moments, shown as numbered sources."""
    hits = retrieval.vector_search(db, question, k=SEARCH_RESULTS, highlight=True)
    if not hits:
        return NO_CONTEXT_ANSWER, []
    citations, lines = [], []
    for n, h in enumerate(hits, start=1):
        citations.append(Citation(n=n, video_id=h["video_id"], title=h["title"], start_sec=h["start_sec"], end_sec=h["end_sec"]))
        excerpt = " ".join((h.get("highlight") or h["text"]).split())
        lines.append(f"[{n}] {h['title']} at {mmss(h['start_sec'])}: “{excerpt[:200]}”")
    return "Top matches:\n" + "\n".join(lines), citations


def memory_text(answer: str, citations: list[Citation]) -> str:
    """The answer as it is remembered: [n] markers mean nothing in a later turn, so they become (title @ mm:ss)."""
    by_n = {c.n: c for c in citations}

    def source(m: re.Match) -> str:
        c = by_n.get(int(m[1]))
        return f" ({c.title} @ {mmss(c.start_sec)})" if c else ""

    return re.sub(r"[ 	]*\[(\d+)\]", source, answer).strip()  # the marker's own leading space is replaced, not doubled


def history_messages(turns: list[Turn]) -> list[Message]:
    out: list[Message] = []
    for t in turns:
        out += [UserMsg(t.question), ModelMsg(text=t.answer)]
    return out


def ask(
    session_factory: Callable[[], Any], user_id: str, question: str, *, mode: str = "rag", session_id: str | None = None,
    llm: LLM | None = None, meter: UsageMeter | None = None, memory: ChatMemory | None = None,
    on_event: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] | None = None,
) -> AskResult:
    meter = meter or UsageMeter()
    memory = memory or ChatMemory()
    emit = on_event or (lambda event: None)

    meter.check(user_id)  # out of tokens for today: refuse BEFORE doing any work (raises DailyBudgetExceeded)
    budget = CallBudget(settings.AGENT_MAX_LLM_CALLS)  # one cap for router + RAG + agent together
    metered = MeteredLLM(llm or get_llm(), budget, meter=meter, user_id=user_id, should_stop=should_stop)
    turns = memory.load(user_id, session_id)

    if mode == "auto":
        decision = router.classify(question, llm=metered, has_history=bool(turns))
    else:
        decision = router.RouteDecision(mode, "requested")
    emit({"type": "route", "route": decision.route, "requested": mode, "reason": decision.reason})

    trace: list[TraceStep] = []
    if decision.route == "agent":
        agent = run_agent(session_factory, question, llm=metered, history=history_messages(turns), on_event=on_event,
                          should_stop=should_stop)
        answer, citations, trace = agent.answer, agent.citations, agent.trace
    elif decision.route == "search":
        with session_factory() as db:
            answer, citations = search_answer(db, question)
    else:
        with session_factory() as db:
            result = rag.answer(db, question, llm=metered)
        answer, citations = result.answer, result.citations

    memory.append(user_id, session_id, question, memory_text(answer, citations))
    log.info("ask: route=%s requested=%s llm_calls=%d tokens=%d steps=%d", decision.route, mode, budget.calls, budget.tokens, len(trace))
    return AskResult(answer=answer, citations=citations, route=decision.route, requested=mode, route_reason=decision.reason,
                     trace=trace, llm_calls=budget.calls, tokens=budget.tokens)
