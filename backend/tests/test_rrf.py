"""Reciprocal Rank Fusion, dedupe-by-rrf and rerank ordering with hand-made rankings. No DB, no models."""
import pytest

from app.core.config import settings
from app.services import rerank
from app.services.search_logic import dedupe_overlapping, rerank_order, rrf_merge


def h(cid, video="v1", start=0.0, end=30.0, **extra):
    return {"chunk_id": cid, "video_id": video, "start_sec": start, "end_sec": end, "text": f"text {cid}", **extra}


def ids(hits):
    return [x["chunk_id"] for x in hits]


def test_rrf_scores_follow_the_formula_with_k_60():
    out = rrf_merge({"vector": [h("a"), h("b"), h("c")], "keyword": [h("c"), h("a")]}, k=60)
    by = {x["chunk_id"]: x for x in out}
    assert by["a"]["rrf"] == pytest.approx(1 / 61 + 1 / 62)  # rank 1 in vector, rank 2 in keyword
    assert by["b"]["rrf"] == pytest.approx(1 / 62)
    assert by["c"]["rrf"] == pytest.approx(1 / 63 + 1 / 61)


def test_a_chunk_both_lists_return_beats_a_chunk_only_one_list_ranks_first():
    out = rrf_merge({"vector": [h("only_v"), h("both")], "keyword": [h("only_k"), h("both")]})
    assert ids(out)[0] == "both"  # 1/62 + 1/62 > 1/61
    assert out[0]["found_by"] == ["vector", "keyword"]


def test_hand_worked_example_order():
    vector = [h("A"), h("B"), h("C"), h("D")]
    keyword = [h("C"), h("E"), h("A")]
    out = rrf_merge({"vector": vector, "keyword": keyword})
    # A: 1/61+1/63 = .032266  C: 1/63+1/61 = .032266 (tie, A's best rank is the same 1: vector wins the tie)
    # B: 1/62 = .016129  E: 1/62 = .016129 (tie: B listed first)  D: 1/64 = .015625
    assert ids(out) == ["A", "C", "B", "E", "D"]
    assert [x["found_by"] for x in out] == [["vector", "keyword"], ["vector", "keyword"], ["vector"], ["keyword"], ["vector"]]


def test_one_empty_list_just_keeps_the_other_order():
    assert ids(rrf_merge({"vector": [h("a"), h("b")], "keyword": []})) == ["a", "b"]
    assert ids(rrf_merge({"vector": [], "keyword": [h("x"), h("y")]})) == ["x", "y"]
    assert rrf_merge({"vector": [], "keyword": []}) == []


def test_k_changes_how_much_top_ranks_dominate():
    rankings = {"vector": [h("a"), h("x"), h("b")], "keyword": [h("y"), h("z"), h("b")]}
    assert rrf_merge(rankings, k=0)[0]["chunk_id"] == "a"  # a: 1/1 = 1.0, b: 1/3 + 1/3 = 0.67: a rank 1 dominates
    assert rrf_merge(rankings, k=1000)[0]["chunk_id"] == "b"  # nearly flat (b: 2/1003 > a: 1/1001): agreement wins


def test_merge_keeps_fields_of_the_first_source_and_fills_gaps_from_the_other():
    out = rrf_merge({"vector": [h("a", score=0.9)], "keyword": [h("a", score=0.1, extra_only="kw")]})
    assert out[0]["score"] == 0.9 and out[0]["extra_only"] == "kw"


def test_inputs_are_not_mutated_and_duplicates_inside_one_list_count_once():
    vector = [h("a"), h("a")]
    out = rrf_merge({"vector": vector, "keyword": [h("a")]})
    assert "rrf" not in vector[0] and len(out) == 1
    assert out[0]["rrf"] == pytest.approx(1 / 61 + 1 / 61)  # vector counted once (its first rank), keyword once


def test_ties_are_deterministic():
    a = rrf_merge({"vector": [h("x"), h("y")], "keyword": [h("y"), h("x")]})
    b = rrf_merge({"vector": [h("x"), h("y")], "keyword": [h("y"), h("x")]})
    assert ids(a) == ids(b) == ["x", "y"]  # same rrf, x's best rank came from the first source


def test_dedupe_can_order_by_rrf_instead_of_score():
    # B overlaps A. Cosine says B is better, fusion says A is: with key="rrf" A must survive.
    a = h("A", start=0, end=30, score=0.5, rrf=0.03)
    b = h("B", start=25, end=55, score=0.9, rrf=0.02)
    assert ids(dedupe_overlapping([a, b], key="rrf")) == ["A"]
    assert ids(dedupe_overlapping([a, b])) == ["B"]  # default unchanged
    far = h("C", start=100, end=130, score=0.1, rrf=0.01)
    assert ids(dedupe_overlapping([b, far, a], key="rrf")) == ["A", "C"]


def test_rerank_order_sorts_by_score_and_keeps_incoming_order_on_ties():
    out = rerank_order([h("a"), h("b"), h("c"), h("d")], [0.1, 2.5, 0.1, -3.0])
    assert ids(out) == ["b", "a", "c", "d"]
    assert out[0]["rerank_score"] == 2.5
    with pytest.raises(ValueError):
        rerank_order([h("a")], [])


def test_rerank_rescored_top_pool_then_keeps_k(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_CANDIDATES", 3)
    seen = {}

    def scorer(q, texts):
        seen["q"], seen["texts"] = q, texts
        return [0.0, 5.0, 1.0]

    hits = [h("a"), h("b"), h("c"), h("d")]
    out = rerank.rerank("question", hits, k=2, scorer=scorer)
    assert ids(out) == ["b", "c"]
    assert seen == {"q": "question", "texts": ["text a", "text b", "text c"]}  # only the top 3 candidates are scored; "d" never is


def test_rerank_failure_falls_back_to_the_incoming_order(monkeypatch):
    def broken(q, texts):
        raise OSError("model download blocked")

    hits = [h("a"), h("b"), h("c")]
    assert ids(rerank.rerank("q", hits, k=2, scorer=broken)) == ["a", "b"]
    assert ids(rerank.rerank("q", [h("only")], k=5, scorer=broken)) == ["only"]  # nothing to reorder: model not needed
