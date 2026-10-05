from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # rag: retrieve once and answer (Phase 8). agent: plan and call tools. auto: let the router choose.
    mode: Literal["rag", "agent", "auto"] = "rag"
    # Client-made id of a chat. The SERVER keeps the last few turns for it, so follow-ups work and history cannot be forged.
    session_id: str | None = Field(None, pattern=SESSION_ID_PATTERN)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("Question is too short")
        return v


class CitationOut(BaseModel):
    n: int
    video_id: str
    title: str
    start_sec: float
    end_sec: float


class TraceStepOut(BaseModel):
    step: int
    index: int
    tool: str
    args: dict[str, Any]
    label: str
    summary: str
    latency_ms: int
    error: bool


class UsageOut(BaseModel):
    llm_calls: int
    tokens: int


class AskOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    mode: Literal["rag", "agent", "auto"]  # what was asked for
    route: Literal["search", "rag", "agent"]  # what was used
    route_reason: str
    trace: list[TraceStepOut] = []
    usage: UsageOut
