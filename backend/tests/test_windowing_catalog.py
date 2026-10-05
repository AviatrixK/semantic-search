"""Transcript windows and the video catalog: pure stitching tests with real chunk text, and real-database tests."""
import uuid
from datetime import date, datetime, timezone

import pytest

from app.models import Chunk, Video
from app.services import catalog, windowing
from app.services.windowing import get_window, stitch_chunks, trim_overlap

# Real chunks 0-2 of the 78-chunk test video: ~30 s windows that overlap by 9-11 s.
C0 = ("Every time you open your mouth and you speak, people unconsciously categorize you into one of these three categories. Are you an "
      "awkward communicator? Are you a pretty good communicator? Or are you a natural communicator? And after 15 years as a communication "
      "coach, training Fortune 500 companies and speaking to millions of people around the world, I've discovered that you can tell which "
      "level someone sits at based on how they use their voice, how they use their body language, and how they use their words.")
C1 = ("I've discovered that you can tell which level someone sits at based on how they use their voice, how they use their body language, "
      "and how they use their words. I'm going to explain the exact behaviors at each level, so by the end you know which category you're "
      "in, and you know how to progress from one category to the next. To make this even more fun, we're going to put these people you see "
      "here on the table into one of these categories, and see if you can guess where they sit on the board.")
C2 = ("To make this even more fun, we're going to put these people you see here on the table into one of these categories, and see if you "
      "can guess where they sit on the board. Well, why is my face there then? Well Craig, it's so we can demonstrate to the audience "
      "there's a level lower than rookie.")
CHUNKS = [{"idx": 0, "start_sec": 0, "end_sec": 28, "text": C0}, {"idx": 1, "start_sec": 18.6, "end_sec": 48, "text": C1},
          {"idx": 2, "start_sec": 37.8, "end_sec": 67, "text": C2}]


def test_trim_overlap_on_real_chunks():
    text, removed = trim_overlap(C0, C1)
    assert removed > 100 and text.startswith("I'm going to explain")


def test_trim_overlap_respects_word_boundaries_and_no_match():
    assert trim_overlap("say hello world", "world is big") == ("is big", 5)
    assert trim_overlap("say hello", "lo there") == ("lo there", 0)
    assert trim_overlap("abc", "xyz") == ("xyz", 0)


def test_stitching_removes_the_repeats_and_keeps_order():
    segs = stitch_chunks(CHUNKS)
    joined = " ".join(s["text"] for s in segs)
    assert joined.count("I've discovered that you can tell") == 1
    assert joined.count("To make this even more fun") == 1
    assert segs[0]["start_sec"] == 0 and segs[1]["start_sec"] == 28 and segs[2]["start_sec"] == 48  # new text starts where the previous chunk ended
    assert segs[2]["text"].startswith("Well, why is my face there then?")
    assert [s["end_sec"] for s in segs] == [28, 48, 67]


def test_stitching_handles_unsorted_input_gaps_and_blank_chunks():
    segs = stitch_chunks([{"idx": 1, "start_sec": 40, "end_sec": 60, "text": "after a gap"},
                          {"idx": 0, "start_sec": 0, "end_sec": 20, "text": "before"},
                          {"idx": 2, "start_sec": 60, "end_sec": 80, "text": "   "}])
    assert [(s["start_sec"], s["text"]) for s in segs] == [(0, "before"), (40, "after a gap")]
    assert stitch_chunks([]) == []


@pytest.fixture
def seeded(api, db_session):
    older = Video(id=uuid.uuid4(), title="Public speaking 101", storage_key="a", status="ready", duration_sec=300,
                  created_at=datetime(2026, 1, 10, 12, tzinfo=timezone.utc))
    newer = Video(id=uuid.uuid4(), title="100% Confidence_talk", storage_key="b", status="ready", duration_sec=1700,
                  created_at=datetime(2026, 3, 4, 0, 0, tzinfo=timezone.utc))
    no_duration = Video(id=uuid.uuid4(), title="Unknown length", storage_key="c", status="ready", duration_sec=None,
                        created_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    processing = Video(id=uuid.uuid4(), title="Still processing", storage_key="d", status="uploaded", duration_sec=100)
    db_session.add_all([older, newer, no_duration, processing])
    db_session.flush()
    db_session.add_all(Chunk(id=uuid.uuid4(), video_id=newer.id, idx=c["idx"], start_sec=c["start_sec"], end_sec=c["end_sec"],
                             text=c["text"], embedding=[1.0] + [0.0] * 383) for c in CHUNKS)
    db_session.commit()
    return {"older": older, "newer": newer, "none": no_duration, "processing": processing}


# ---- catalog (real SQL)
def titles(db, **kw):
    return [v["title"] for v in catalog.list_ready_videos(db, **kw)]


def test_only_ready_videos_newest_first(seeded, db_session):
    assert titles(db_session) == ["100% Confidence_talk", "Unknown length", "Public speaking 101"]


def test_uploaded_after_is_inclusive_from_midnight_utc(seeded, db_session):
    assert titles(db_session, uploaded_after=date(2026, 3, 4)) == ["100% Confidence_talk"]  # created exactly 00:00 UTC that day
    assert titles(db_session, uploaded_after=date(2026, 3, 5)) == []
    assert titles(db_session, uploaded_after=date(2026, 1, 11)) == ["100% Confidence_talk", "Unknown length"]


def test_max_duration_excludes_longer_and_unknown_lengths(seeded, db_session):
    assert titles(db_session, max_duration_sec=600) == ["Public speaking 101"]
    assert titles(db_session, max_duration_sec=2000) == ["100% Confidence_talk", "Public speaking 101"]


def test_title_filter_is_case_insensitive_and_treats_wildcards_literally(seeded, db_session):
    assert titles(db_session, title_contains="SPEAKING") == ["Public speaking 101"]
    assert titles(db_session, title_contains="100%") == ["100% Confidence_talk"]  # % is not a wildcard
    assert titles(db_session, title_contains="e_t") == ["100% Confidence_talk"]  # _ is not a wildcard: no "e_t" match elsewhere
    assert titles(db_session, title_contains="%") == ["100% Confidence_talk"]
    assert titles(db_session, title_contains="zzz") == []


def test_filters_combine_and_the_limit_applies(seeded, db_session):
    assert titles(db_session, uploaded_after=date(2026, 1, 1), max_duration_sec=600, title_contains="public") == ["Public speaking 101"]
    assert len(catalog.list_ready_videos(db_session, limit=2)) == 2
    row = catalog.list_ready_videos(db_session, title_contains="Confidence")[0]
    assert row["uploaded"] == "2026-03-04" and row["duration_sec"] == 1700 and uuid.UUID(row["video_id"])


# ---- windows (real SQL)
def test_a_window_returns_overlapping_chunks_stitched_without_repeats(seeded, db_session):
    win = get_window(db_session, seeded["newer"].id, 20, 60)
    assert win["title"] == "100% Confidence_talk" and win["video_id"] == str(seeded["newer"].id)
    text = " ".join(s["text"] for s in win["segments"])
    assert len(win["segments"]) == 3 and text.count("I've discovered that you can tell") == 1


def test_a_narrow_window_only_returns_the_chunks_that_touch_it(seeded, db_session):
    win = get_window(db_session, seeded["newer"].id, 50, 60)
    assert [s["end_sec"] for s in win["segments"]] == [67]  # only chunk 2 reaches 50-60 (chunk 1 ends at 48)
    assert get_window(db_session, seeded["newer"].id, 500, 560)["segments"] == []


def test_a_window_of_an_unknown_video_is_none(seeded, db_session):
    assert get_window(db_session, uuid.uuid4(), 0, 60) is None
    assert windowing.get_window is get_window
