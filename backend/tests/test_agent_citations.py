"""Citation validation: the agent may cite only what its tools really returned."""
import pytest

from app.agent.citations import Evidence, fallback_answer, validate_citations
from app.services.rag import NO_CONTEXT_ANSWER

A = "8f27bfee-4387-448e-bc0f-29cb05d75f80"
B = "5d1a0c42-9b7e-4a53-8a0f-1c2d3e4f5a6b"


def evidence():
    e = Evidence()
    e.add_result({"clips": [
        {"video_id": A, "title": "Video A", "start_sec": 100.0, "end_sec": 130.0},
        {"video_id": A, "title": "Video A", "start_sec": 283.4, "end_sec": 313.4},
        {"video_id": B, "title": "Video B", "start_sec": 40.0, "end_sec": 70.0}]})
    e.add_result({"video_id": A, "title": "Video A", "segments": [{"start_sec": 500.0, "end_sec": 560.0, "text": "w"}]})
    return e


def test_valid_markers_become_numbered_citations_in_order_of_first_use():
    text, cites = validate_citations(f"B says [{B}@40]. A says [{A}@100] and again [{B}@40].", evidence())
    assert text == "B says [1]. A says [2] and again [1]."
    assert [(c.n, c.video_id, c.title, c.start_sec, c.end_sec) for c in cites] == [
        (1, B, "Video B", 40.0, 70.0), (2, A, "Video A", 100.0, 130.0)]


def test_a_video_the_tools_never_returned_is_removed():
    text, cites = validate_citations(f"Real [{A}@100]. Invented [11111111-2222-3333-4444-555555555555@10].", evidence())
    assert text == "Real [1]. Invented." and [c.video_id for c in cites] == [A]


def test_a_time_outside_every_retrieved_range_is_removed():
    text, cites = validate_citations(f"In range [{A}@110], out of range [{A}@1000], before [{A}@10].", evidence())
    assert text == "In range [1], out of range, before." and len(cites) == 1


def test_one_second_of_tolerance_for_rounded_refs_but_not_two():
    e = evidence()
    assert validate_citations(f"x [{A}@283]", e)[1], "ref is int(283.4): must match the clip starting at 283.4"
    assert validate_citations(f"x [{A}@99]", e)[1], "1s before a range start is tolerated"
    assert not validate_citations(f"x [{A}@98]", e)[1]
    assert validate_citations(f"x [{A}@131]", e)[1], "1s after a range end is tolerated"
    assert not validate_citations(f"x [{A}@132]", e)[1]


def test_window_segments_are_evidence_too():
    text, cites = validate_citations(f"From the window [{A}@530].", evidence())
    assert text == "From the window [1]." and cites[0].start_sec == 530.0 and cites[0].end_sec == 560.0


def test_comma_lists_and_decimals_and_an_s_suffix():
    text, cites = validate_citations(f"Both agree [{A}@100, {B}@40] and [{A}@283.9s].", evidence())
    assert text == "Both agree [1][2] and [3]."
    assert [c.n for c in cites] == [1, 2, 3]


def test_mixed_valid_and_invalid_in_one_group():
    text, cites = validate_citations(f"Mixed [{A}@100; {A}@9999].", evidence())
    assert text == "Mixed [1]." and len(cites) == 1


def test_ids_are_case_insensitive_and_unique_prefixes_are_accepted():
    e = evidence()
    assert validate_citations(f"x [{A.upper()}@100]", e)[0] == "x [1]"
    assert validate_citations("x [8f27bfee@100]", e)[0] == "x [1]"  # shortened by the model: unambiguous, so accepted
    assert validate_citations("x [8f27@100]", e)[1] == []  # too short to trust


def test_an_ambiguous_prefix_is_rejected():
    e = Evidence()
    e.add_result({"clips": [{"video_id": "aaaaaaaa-0000-0000-0000-000000000001", "title": "1", "start_sec": 0, "end_sec": 10},
                            {"video_id": "aaaaaaaa-0000-0000-0000-000000000002", "title": "2", "start_sec": 0, "end_sec": 10}]})
    assert validate_citations("x [aaaaaaaa@5]", e)[1] == []
    assert validate_citations("x [aaaaaaaa-0000-0000-0000-000000000002@5]", e)[1][0].title == "2"


def test_text_without_markers_and_non_citation_brackets_are_left_alone():
    e = evidence()
    assert validate_citations("Plain answer.", e) == ("Plain answer.", [])
    assert validate_citations("He said [sic] and [1] and [a b@3] and [x].", e)[0] == "He said [sic] and [1] and [a b@3] and [x]."


def test_an_empty_evidence_set_makes_every_marker_invalid():
    text, cites = validate_citations(f"Claim [{A}@100].", Evidence())
    assert text == "Claim." and cites == []


def test_gaps_left_by_removed_markers_are_tidied():
    text, _ = validate_citations(f"Made up [{A}@9999] and also [{B}@9999], then [{A}@9999].", evidence())
    assert text == "Made up and also, then."


def test_results_without_clips_add_no_evidence():
    e = Evidence()
    e.add_result({"count": 0, "clips": [], "note": "nothing"})
    e.add_result({"error": "bad"})
    e.add_result({"videos": [{"video_id": A, "title": "Only listed"}]})  # a listing proves a video exists, not any moment of it
    assert e.ranges == []


def test_fallback_answer_lists_the_best_moments_and_survives_validation():
    e = evidence()
    text, cites = validate_citations(fallback_answer(e), e)
    assert text.startswith("I could not finish my research")
    assert "[1]" in text and "[2]" in text and "[3]" in text and len(cites) == 3
    assert fallback_answer(Evidence()) == NO_CONTEXT_ANSWER
    assert fallback_answer(e, limit=1).count("[") == 1
