"""Decides HOW to answer: "search" (a keyword lookup: just show matching moments, no LLM), "rag" (one fact: retrieve once,
one LLM call) or "agent" (comparison, several parts, filters: tools and several LLM calls). Heuristics decide almost
everything for free; a tiny LLM classification is used only for the genuinely ambiguous middle."""
import logging
import re
from dataclasses import dataclass

from app.agent.prompts import ROUTER_SYSTEM
from app.services.llm import LLM

log = logging.getLogger(__name__)
ROUTES = ("search", "rag", "agent")

QUESTION_STARTS = frozenset("""what why how when where who whom whose which is are was were do does did can could should would will
shall explain summarize summarise describe tell give show list find define""".split())

# Words that mean "this needs more than one search or a filter": comparisons, several parts, dates, lengths, ordinals.
AGENT_PATTERNS = [
    (re.compile(r"\b(compare[sd]?|comparison|versus|vs\.?|differences?|differ(?:s|ent)?|contrast|similarit(?:y|ies)|"
                r"both videos|each video|every video|all (?:the )?videos|across (?:the )?videos|between (?:the )?(?:two )?videos?)\b"),
     "comparison"),
    (re.compile(r"\b(and also|as well as|and then|first\b.+\bthen|step by step|pros and cons)\b"), "several parts"),
    (re.compile(r"\b(uploaded|upload date|recent(?:ly)?|latest|newest|oldest|yesterday|today|"
                r"last (?:week|month|year)|this (?:week|month|year))\b"), "date or recency filter"),
    (re.compile(r"\b20\d\d-\d\d-\d\d\b"), "date filter"),
    (re.compile(r"\b(shorter|longer) than\b|\b(?:under|over|less than|more than)\s+\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?)\b"),
     "length filter"),
    (re.compile(r"\b(first|second|third|fourth|fifth|last|other|another)\s+(video|one|clip|talk)\b"), "refers to a specific video"),
    (re.compile(r"\bhow many (videos|talks|clips)\b"), "asks about the library"),
]
FOLLOW_UP_START = re.compile(r"^(and|but|also|so|what about|how about|why|then)\b")
ANAPHORA = re.compile(r"\b(it|that|this|they|them|those|these|he|she|his|her|the same|more|else)\b")


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    used_llm: bool = False


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text)


def classify_heuristic(question: str, *, has_history: bool = False) -> RouteDecision | None:
    """A decision, or None when the input is ambiguous."""
    q = " ".join(question.lower().split())
    words = _words(q)
    n = len(words)

    for pattern, reason in AGENT_PATTERNS:
        if pattern.search(q):
            return RouteDecision("agent", reason)
    if q.count("?") >= 2:
        return RouteDecision("agent", "several questions")
    if n > 30:
        return RouteDecision("agent", "long, multi-part question")
    if has_history and n <= 10 and (FOLLOW_UP_START.match(q) or (n <= 8 and ANAPHORA.search(q))):
        return RouteDecision("agent", "follow-up to the previous question")

    question_like = q.endswith("?") or (n > 0 and words[0] in QUESTION_STARTS)
    if question_like:
        return RouteDecision("rag", "single question") if n <= 25 else RouteDecision("agent", "long question")
    if len(q) >= 2 and q[0] in "\"'“" and q[-1] in "\"'”" and n <= 8:
        return RouteDecision("search", "quoted phrase")
    if n <= 3:
        return RouteDecision("search", "short keyword lookup")
    if n >= 7:
        return RouteDecision("rag", "descriptive request")
    return None  # 4-6 words, not phrased as a question: could be a lookup or a request


def classify(question: str, *, llm: LLM | None = None, has_history: bool = False) -> RouteDecision:
    decided = classify_heuristic(question, has_history=has_history)
    if decided is not None:
        return decided
    if llm is None:
        return RouteDecision("rag", "ambiguous, defaulted to a direct answer")
    try:
        reply = llm.generate(system=ROUTER_SYSTEM, prompt=question).text
    except Exception as exc:  # budget, outage, anything: routing must never make a request fail
        log.warning("router: LLM classification failed (%s), defaulting to rag", type(exc).__name__)
        return RouteDecision("rag", "ambiguous, classifier unavailable")
    first = (re.findall(r"[a-z]+", reply.lower()) or [""])[0]
    if first in ROUTES:
        return RouteDecision(first, "ambiguous, classified by the model", used_llm=True)
    return RouteDecision("rag", "ambiguous, unclear classifier reply", used_llm=True)
