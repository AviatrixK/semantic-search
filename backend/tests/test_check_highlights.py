from app.scripts.check_highlights import classify

TWO = "Hello everyone. Robots can teach us how to live."


def test_single_sentence_chunks_need_no_rows():
    assert classify("Just one sentence here", []) == "single"


def test_multi_sentence_chunk_is_stored_missing_or_stale():
    assert classify(TWO, ["Hello everyone.", "Robots can teach us how to live."]) == "stored"
    assert classify(TWO, []) == "missing"
    assert classify(TWO, ["old split rule"]) == "stale"


def test_hosted_providers_have_no_sentence_vectors_to_check_or_backfill(capsys):
    from unittest.mock import patch

    from app.scripts import backfill_sentences, check_highlights
    from app.services import embedders
    hosted = embedders.GeminiEmbedder(model="m", dim=8, api_key="k")
    boom = patch("app.scripts.check_highlights.SessionLocal", side_effect=AssertionError("the database must not be touched"))
    boom2 = patch("app.scripts.backfill_sentences.SessionLocal", side_effect=AssertionError("the database must not be touched"))
    with patch.object(embedders, "get_embedder", return_value=hosted), boom, boom2:
        assert check_highlights.main() == 0 and backfill_sentences.main() == 0
    assert capsys.readouterr().out.count("highlights are lexical") == 2
