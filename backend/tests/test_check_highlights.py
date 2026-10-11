from app.scripts.check_highlights import classify

TWO = "Hello everyone. Robots can teach us how to live."


def test_single_sentence_chunks_need_no_rows():
    assert classify("Just one sentence here", []) == "single"


def test_multi_sentence_chunk_is_stored_missing_or_stale():
    assert classify(TWO, ["Hello everyone.", "Robots can teach us how to live."]) == "stored"
    assert classify(TWO, []) == "missing"
    assert classify(TWO, ["old split rule"]) == "stale"
