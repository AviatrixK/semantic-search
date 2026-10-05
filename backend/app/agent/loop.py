"""The agent loop, hand-rolled. No framework: a plain `for` loop around the model.

    ask the model  ->  did it call tools?  -- no --> that text is the answer, done
                                           -- yes -> run the tools, give the results back, ask the model again

Every pass of the loop is a "step". After max_steps steps of tool use the model is made to answer with what it has.
The model decides WHAT to do; this code only runs the tools, enforces limits, keeps the evidence and checks the citations."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agent.citations import Evidence, fallback_answer, validate_citations
from app.agent.limits import Cancelled, CallCapExceeded
from app.agent.prompts import FORCE_FINAL_MESSAGE, SYSTEM_PROMPT
from app.agent.tools import (TOOL_SPECS, ToolOutcome, describe_call, execute_tool, summarize_outcome, truncate_result)
from app.core.config import settings
from app.services.llm import LLM, Message, ModelMsg, ToolCall, ToolMsg, ToolResponse, UserMsg, get_llm
from app.services.rag import Citation
from app.services.usage import DailyBudgetExceeded

log = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 4  # tool calls executed together in one step; extra calls get an error back
MAX_ARG_CHARS = 200  # long string arguments are shortened in the trace


@dataclass(frozen=True)
class TraceStep:
    step: int  # which model round (1-based); parallel calls of one round share it
    index: int  # position among that round's calls
    tool: str
    args: dict
    label: str  # "Searching “x”": shown while the call runs
    summary: str  # "Found 5 clips": shown when it is done
    latency_ms: int
    error: bool


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation]
    trace: list[TraceStep]
    steps: int  # rounds in which the model used tools
    llm_calls: int
    stopped: str  # "answer" | "max_steps" | "budget" (call cap or daily token budget)


def _shorten_args(args: Any) -> dict:
    if not isinstance(args, dict):
        return {}
    return {k: (v[:MAX_ARG_CHARS] + "…" if isinstance(v, str) and len(v) > MAX_ARG_CHARS else v) for k, v in args.items()}


def run_agent(
    session_factory: Callable[[], Any], question: str, *, llm: LLM | None = None, max_steps: int | None = None,
    history: list[Message] | None = None, on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None, tool_token_budget: int | None = None,
) -> AgentResult:
    llm = llm or get_llm()
    max_steps = settings.AGENT_MAX_STEPS if max_steps is None else max_steps
    emit = on_event or (lambda event: None)
    stop = should_stop or (lambda: False)

    messages: list[Message] = [*(history or []), UserMsg(question)]
    evidence = Evidence()
    trace: list[TraceStep] = []
    llm_calls = 0
    steps_with_tools = 0
    stopped = "answer"
    final_text: str | None = None

    def run_one(step: int, index: int, call: ToolCall, skipped: bool) -> tuple[ToolOutcome, TraceStep]:
        label = describe_call(call.name, call.args)
        emit({"type": "step_start", "step": step, "index": index, "tool": call.name, "args": _shorten_args(call.args), "label": label})
        started = time.monotonic()
        if skipped:
            outcome = ToolOutcome({"error": f"Too many tool calls in one step (maximum {MAX_PARALLEL_CALLS}). Call this one again in the next step."}, False)
        else:
            outcome = execute_tool(session_factory, call.name, call.args)
        latency = int((time.monotonic() - started) * 1000)
        outcome = ToolOutcome(truncate_result(outcome.result, tool_token_budget), outcome.ok)  # what the model sees is what counts as evidence
        summary = summarize_outcome(call.name, call.args, outcome)
        step_trace = TraceStep(step, index, call.name, _shorten_args(call.args), label, summary, latency, not outcome.ok)
        emit({"type": "step_result", "step": step, "index": index, "tool": call.name, "summary": summary, "latency_ms": latency,
              "error": not outcome.ok})
        return outcome, step_trace

    try:
        for step in range(1, max_steps + 1):
            if stop():
                raise Cancelled()
            turn = llm.generate_turn(system=SYSTEM_PROMPT, messages=messages, tools=TOOL_SPECS)
            llm_calls += 1
            if not turn.tool_calls:  # the model is done researching: its text is the answer
                final_text = turn.text
                break
            steps_with_tools += 1
            messages.append(ModelMsg(text=turn.text, tool_calls=tuple(turn.tool_calls), raw=turn.raw))
            calls = list(turn.tool_calls)
            if len(calls) == 1:
                results = [run_one(step, 0, calls[0], False)]
            else:  # several calls in one step are independent: run them at the same time
                with ThreadPoolExecutor(max_workers=min(len(calls), MAX_PARALLEL_CALLS)) as pool:
                    futures = [pool.submit(run_one, step, i, c, i >= MAX_PARALLEL_CALLS) for i, c in enumerate(calls)]
                    results = [f.result() for f in futures]  # same order as the calls
            for outcome, step_trace in results:
                evidence.add_result(outcome.result)
                trace.append(step_trace)
            messages.append(ToolMsg(tuple(ToolResponse(name=c.name, content=o.result, id=c.id) for c, (o, _t) in zip(calls, results))))
        else:  # max_steps rounds and the model still wants more tools: make it answer now with what it has
            stopped = "max_steps"
            if stop():
                raise Cancelled()
            messages.append(UserMsg(FORCE_FINAL_MESSAGE))
            turn = llm.generate_turn(system=SYSTEM_PROMPT, messages=messages, tools=TOOL_SPECS, allow_tools=False)
            llm_calls += 1
            final_text = turn.text
    except (CallCapExceeded, DailyBudgetExceeded) as exc:  # out of allowed model calls or tokens: finish gracefully
        stopped = "budget"
        log.warning("agent stopped early: %s", exc)

    if not final_text or not final_text.strip():
        final_text = fallback_answer(evidence)
    answer, citations = validate_citations(final_text, evidence)
    return AgentResult(answer=answer, citations=citations, trace=trace, steps=steps_with_tools, llm_calls=llm_calls, stopped=stopped)
