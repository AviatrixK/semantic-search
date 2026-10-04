"""The router: heuristics decide almost everything for free; the LLM is used only for the ambiguous middle."""
import pytest

from app.agent import router
from app.agent.prompts import ROUTER_SYSTEM
from app.agent.router import classify, classify_heuristic
from app.services.llm import LLMUnavailable
from tests.fakes import FakeLLM

SEARCH = ["public speaking", "voice body language", "pauses", "stage fright", '"pause between ideas"', "Eye contact"]
RAG = ["How do rookie communicators use their voice?", "What is unconscious incompetence", "Explain the three levels of communicators",
       "Why do pauses matter in a talk", "Summarize the section on body language.", "Is eye contact important?"]
AGENT = [
    ("Compare what video A and B say about confidence", "comparison"),
    ("What is the difference between rookie and natural speakers?", "comparison"),
    ("Which videos were uploaded last week?", "date or recency filter"),
    ("Find videos uploaded after 2026-03-01 about fear", "date or recency filter"),
    ("Videos shorter than 10 minutes about voice", "length filter"),
    ("videos under 20 minutes on pacing", "length filter"),
    ("What does the second video say about gestures?", "refers to a specific video"),
    ("How many videos mention fear?", "asks about the library"),
    ("What do they say about pacing? And about gestures?", "several questions"),
    ("List the key points and also the common mistakes", "several parts"),
    ("What are the pros and cons of using notes?", "several parts"),
    ("Summarize all the videos", "comparison"),
]
AMBIGUOUS = ["tips for a nervous speaker", "sounding confident on stage while talking", "speaker advice pauses rhythm tone"]


@pytest.mark.parametrize("q", SEARCH)
def test_short_keyword_lookups_go_to_search(q):
    d = classify_heuristic(q)
    assert d is not None and d.route == "search"


@pytest.mark.parametrize("q", RAG)
def test_single_questions_go_to_rag(q):
    d = classify_heuristic(q)
    assert d is not None and d.route == "rag", q


@pytest.mark.parametrize("q,reason", AGENT)
def test_comparisons_filters_and_multi_part_questions_go_to_the_agent(q, reason):
    d = classify_heuristic(q)
    assert (d.route, d.reason) == ("agent", reason), q


def test_a_very_long_question_goes_to_the_agent():
    q = "What does the speaker say about " + " and ".join(["pacing"] * 3) + " " + " ".join(["really"] * 30) + "?"
    assert classify_heuristic(q).route == "agent"


@pytest.mark.parametrize("q", AMBIGUOUS)
def test_four_to_six_word_non_questions_are_ambiguous(q):
    assert classify_heuristic(q) is None


def test_seven_or_more_words_without_a_question_are_a_descriptive_request():
    assert classify_heuristic("the speaker's advice about pauses and rhythm in presentations").route == "rag"


def test_follow_ups_go_to_the_agent_only_when_there_is_history():
    for q in ["and body language?", "why is that?", "also the gestures part", "what about the tone?", "more on that"]:
        assert classify_heuristic(q, has_history=True).route == "agent", q
        assert classify_heuristic(q, has_history=False) is None or classify_heuristic(q, has_history=False).route != "agent", q


def test_a_self_contained_long_question_is_not_mistaken_for_a_follow_up():
    q = "How do rookie communicators typically use their voice when they are nervous on stage?"
    assert classify_heuristic(q, has_history=True).route == "rag"


# ---- the LLM fallback
@pytest.mark.parametrize("q", SEARCH + RAG + [a for a, _ in AGENT])
def test_decided_inputs_never_call_the_llm(q):
    llm = FakeLLM("agent")
    classify(q, llm=llm)
    assert llm.calls == []


@pytest.mark.parametrize("reply,route", [("agent", "agent"), ("Search.", "search"), ("  RAG\n", "rag"), ("agent - it compares things", "agent")])
def test_ambiguous_input_is_classified_by_the_model(reply, route):
    llm = FakeLLM(reply)
    d = classify("tips for a nervous speaker", llm=llm)
    assert (d.route, d.used_llm) == (route, True)
    assert llm.calls == [{"system": ROUTER_SYSTEM, "prompt": "tips for a nervous speaker"}]


@pytest.mark.parametrize("reply", ["I think it depends", "", "42", "maybe-agent-ish... or not"])
def test_an_unusable_model_reply_falls_back_to_rag(reply):
    d = classify("tips for a nervous speaker", llm=FakeLLM(reply))
    assert d.route == "rag" and d.used_llm is True and "unclear" in d.reason


def test_a_failing_classifier_never_fails_the_request():
    d = classify("tips for a nervous speaker", llm=FakeLLM(LLMUnavailable("down")))
    assert (d.route, d.used_llm) == ("rag", False) and "unavailable" in d.reason
    assert classify("tips for a nervous speaker", llm=None).route == "rag"


def test_history_is_passed_to_the_heuristics():
    assert classify("and body language?", has_history=True).route == "agent"
    assert classify("and body language?", has_history=False).route == "rag"


def test_routes_are_the_three_documented_ones():
    assert router.ROUTES == ("search", "rag", "agent")
