"""Sentence-level embeddings used for search highlights, computed once at ingest (or by the backfill script)
instead of at query time: embedding ~80 sentences per search took ~500 ms on CPU, a stored-vector lookup takes ~ms."""
import uuid

from app.models import ChunkSentence
from app.services import embedding
from app.services.search_logic import split_sentences


def build_rows(chunks: list[tuple[uuid.UUID, str]]) -> list[ChunkSentence]:
    """Rows for every chunk that has more than one sentence (single-sentence chunks need no highlight lookup:
    the sentence is the whole chunk). All sentences go through the model in one batch."""
    plan = []
    for chunk_id, text in chunks:
        sentences = split_sentences(text)
        if len(sentences) > 1:
            plan.extend((chunk_id, i, s) for i, s in enumerate(sentences))
    if not plan:
        return []
    vectors = embedding.embed_batch([s for _, _, s in plan])
    return [ChunkSentence(chunk_id=c, idx=i, text=s, embedding=v) for (c, i, s), v in zip(plan, vectors)]
