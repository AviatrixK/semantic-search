from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

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


class AskOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    mode: Literal["rag"] = "rag"
