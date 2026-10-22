import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import settings
from app.core.db import Base


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String)
    storage_key: Mapped[str] = mapped_column(String)
    duration_sec: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="uploaded")
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    # SHA-256 (hex) of the file as uploaded. Unique where set: the same file cannot be stored twice. NULL for videos that
    # predate the column until `python -m app.scripts.backfill_sha256` has run.
    sha256: Mapped[str | None] = mapped_column(String, unique=True)
    language_hint: Mapped[str | None] = mapped_column(String)  # language chosen on upload (Sarvam code); None = automatic
    transcriber: Mapped[str | None] = mapped_column(String)  # engine behind the current chunks: whisper | sarvam (None = whisper)
    # The Sarvam batch job for this video. Kept so that a retry resumes polling it instead of starting (and paying for) a second one.
    # status: created | uploaded | started | accepted/pending/running/completed/failed (Sarvam's) | consumed (its transcript was used)
    sarvam_job_id: Mapped[str | None] = mapped_column(String)
    sarvam_job_status: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    stage: Mapped[str] = mapped_column(String, default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("video_id", "idx", name="chunks_video_id_idx_key"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    idx: Mapped[int] = mapped_column(Integer)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    text_native: Mapped[str | None] = mapped_column(Text)  # own-script text (equals `text`)
    text_roman: Mapped[str | None] = mapped_column(Text)  # Latin-script transcription (Sarvam only)
    text_en: Mapped[str | None] = mapped_column(Text)  # English translation (Sarvam only)
    embedding = mapped_column(Vector(settings.EMBED_DIM), nullable=True)  # local model only; hosted providers use chunk_embeddings
    # `tsv` is a generated column in the DB; not mapped here so SQLAlchemy never writes it.


class ChunkSentence(Base):
    __tablename__ = "chunk_sentences"
    __table_args__ = (UniqueConstraint("chunk_id", "idx", name="chunk_sentences_chunk_id_idx_key"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    idx: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(settings.EMBED_DIM), nullable=False)


class ChunkEmbedding(Base):
    """A vector of one text variant of a chunk from one embedding model. `model` is "<provider>/<model>@<dimensions>"."""
    __tablename__ = "chunk_embeddings"
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    model: Mapped[str] = mapped_column(String, primary_key=True)
    field: Mapped[str] = mapped_column(String, primary_key=True)  # "native" | "en"
    embedding = mapped_column(Vector(), nullable=False)  # no fixed size: each model has its own
