import random

import pytest

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


def test_no_overlap():
    segs = [seg(i * 10, i * 10 + 10, f"s{i}") for i in range(9)]
    chunks = window(segs, size=30, overlap=0)
    assert [(c.start, c.end) for c in chunks] == [(0, 30), (30, 60), (60, 90)]


def test_invalid_params_rejected():
    for size, overlap in [(0, 0), (30, 30), (30, -1), (10, 20)]:
        with pytest.raises(ValueError):
            window([seg(0, 1, "x")], size=size, overlap=overlap)


def test_blank_segments_dropped_and_text_stripped():
    chunks = window([seg(0, 5, "  hi  "), seg(5, 6, "   "), seg(6, 7, ""), seg(7, 9, "there")], size=30)
    assert len(chunks) == 1 and chunks[0].text == "hi there"
    assert window([seg(0, 1, " "), seg(1, 2, "")]) == []


def test_input_not_mutated():
    segs = [seg(0, 5, " a ")]
    window(segs)
    assert segs == [seg(0, 5, " a ")]


def test_defaults_come_from_settings():
    segs = [seg(i * 10, i * 10 + 10, f"s{i}") for i in range(10)]
    assert window(segs) == window(segs, size=30, overlap=5)


@pytest.mark.parametrize("size,overlap", [(30, 0), (30, 5), (30, 29.9), (10, 9), (1, 0.5)])
def test_invariants_on_random_input(size, overlap):
    rng = random.Random(size * 100 + overlap)
    t, segs = 0.0, []
    for k in range(200):
        t += rng.choice([0, 0, 0.5, 3])  # silence gaps
        d = rng.choice([0.5, 2, 4, 8, 45])  # includes segments longer than the window
        segs.append(seg(t, t + d, f"s{k}"))
        t += d
    chunks = window(segs, size=size, overlap=overlap)  # must terminate
    assert all(a.start < b.start for a, b in zip(chunks, chunks[1:]))  # time order, always progressing
    assert all(a.end <= b.end for a, b in zip(chunks, chunks[1:]))
    for s in segs:  # every segment is fully inside at least one chunk
        assert any(c.start <= s["start"] and s["end"] <= c.end and s["text"] in c.text.split() for c in chunks), s
