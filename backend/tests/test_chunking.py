from app.services.chunking import window


def seg(s, e, t):
    return {"start": s, "end": e, "text": t}


def test_windows_cover_everything_and_overlap():
    segs = [seg(i * 10, i * 10 + 10, f"s{i}") for i in range(10)]  # 0..100s
    chunks = window(segs, size=30, overlap=5)
    assert chunks[0].start == 0 and chunks[-1].end == 100
    assert all(c.end - c.start <= 30 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlapping
        assert b.start > a.start  # always progresses


def test_single_long_segment_still_emitted():
    chunks = window([seg(0, 90, "long")], size=30)
    assert len(chunks) == 1 and chunks[0].text == "long"


def test_empty():
    assert window([]) == []
