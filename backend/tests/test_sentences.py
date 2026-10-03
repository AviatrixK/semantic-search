import uuid
from unittest.mock import patch

from sqlalchemy import text

from app.models import Chunk, ChunkSentence, Video
from app.scripts import backfill_sentences
from app.services import embedding, sentences

DIM = 384


def fake_embed(texts):
    return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


def test_build_rows_skips_single_sentence_chunks_and_embeds_in_one_batch():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with patch.object(embedding, "embed_batch", side_effect=fake_embed) as model:
        rows = sentences.build_rows([(a, "One. Two. Three."), (b, "Just one sentence here"), (c, "Four? Five!")])
    assert model.call_count == 1 and len(model.call_args[0][0]) == 5
    assert [(r.chunk_id, r.idx, r.text) for r in rows] == [
        (a, 0, "One."), (a, 1, "Two."), (a, 2, "Three."), (c, 0, "Four?"), (c, 1, "Five!")]


def test_build_rows_makes_no_model_call_when_nothing_to_embed():
    with patch.object(embedding, "embed_batch", side_effect=fake_embed) as model:
        assert sentences.build_rows([(uuid.uuid4(), "no punctuation at all")]) == []
        assert sentences.build_rows([]) == []
    model.assert_not_called()


def test_backfill_fills_only_chunks_without_sentences_and_is_rerunnable(api, db_session):
    v = Video(id=uuid.uuid4(), title="t", storage_key="raw/x.mp4")
    db_session.add(v)
    db_session.flush()
    chunks = [Chunk(id=uuid.uuid4(), video_id=v.id, idx=i, start_sec=i * 30, end_sec=i * 30 + 30, text=t,
                    embedding=[1.0] + [0.0] * (DIM - 1))
              for i, t in enumerate(["First. Second.", "Only one sentence", "Third. Fourth. Fifth."])]
    db_session.add_all(chunks)
    db_session.flush()
    with patch.object(embedding, "embed_batch", side_effect=fake_embed):
        db_session.add_all(sentences.build_rows([(chunks[2].id, chunks[2].text)]))  # chunk 2 already done
    db_session.commit()

    with patch.object(embedding, "embed_batch", side_effect=fake_embed) as model:
        assert backfill_sentences.main() == 2  # only chunk 0 (two sentences) was missing
        assert model.call_count == 1
        assert backfill_sentences.main() == 0  # nothing left to do
    assert db_session.scalar(text("SELECT count(*) FROM chunk_sentences")) == 5


def test_backfill_rebuilds_stale_rows_after_a_splitter_change(api, db_session):
    v = Video(id=uuid.uuid4(), title="t", storage_key="raw/x.mp4")
    db_session.add(v)
    db_session.flush()
    text_ = "Every time you speak, people categorize you. Are you awkward? Or are you natural?"
    chunk = Chunk(id=uuid.uuid4(), video_id=v.id, idx=0, start_sec=0, end_sec=28, text=text_,
                  embedding=[1.0] + [0.0] * (DIM - 1))
    single = Chunk(id=uuid.uuid4(), video_id=v.id, idx=1, start_sec=28, end_sec=56, text="One sentence only",
                   embedding=[1.0] + [0.0] * (DIM - 1))
    db_session.add_all([chunk, single])
    db_session.flush()
    old_rows = [ChunkSentence(chunk_id=chunk.id, idx=0, text="Every time you speak, people categorize", embedding=[1.0] + [0.0] * (DIM - 1)),
                ChunkSentence(chunk_id=chunk.id, idx=1, text="you. Are you awkward?", embedding=[1.0] + [0.0] * (DIM - 1)),
                ChunkSentence(chunk_id=single.id, idx=0, text="left over", embedding=[1.0] + [0.0] * (DIM - 1))]
    db_session.add_all(old_rows)
    db_session.commit()

    with patch.object(embedding, "embed_batch", side_effect=fake_embed):
        assert backfill_sentences.main() == 3
        assert backfill_sentences.main() == 0  # now consistent with the splitter
    db_session.expire_all()
    got = db_session.execute(text("SELECT text FROM chunk_sentences ORDER BY idx")).scalars().all()
    assert got == ["Every time you speak, people categorize you.", "Are you awkward?", "Or are you natural?"]
