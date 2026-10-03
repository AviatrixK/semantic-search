"""Pure-function tests: no DB, no Redis, no embedding model."""
import copy
import random
import re

import numpy as np
import pytest

from app.services.search_logic import (_DANGLING, _bare, apply_threshold, best_index, dedupe_overlapping,
                                       split_sentences)


def hit(start, end, score, video="v1"):
    return {"video_id": video, "start_sec": start, "end_sec": end, "score": score}


def spans(hits):
    return [(h["video_id"], h["start_sec"], h["end_sec"]) for h in hits]


# ---- threshold
def test_threshold_drops_weak_hits_and_keeps_boundary():
    hits = [hit(0, 30, 0.9), hit(30, 60, 0.25), hit(60, 90, 0.2499), hit(90, 120, -0.1)]
    assert [h["score"] for h in apply_threshold(hits, 0.25)] == [0.9, 0.25]


def test_threshold_edge_cases():
    assert apply_threshold([], 0.25) == []
    assert apply_threshold([hit(0, 1, 0.1)], 0.25) == []  # nonsense query: nothing survives
    assert len(apply_threshold([hit(0, 1, -0.5), hit(1, 2, 0.0)], -1.0)) == 2  # threshold low enough keeps all
    assert apply_threshold([hit(0, 1, 0.3), hit(2, 3, 0.1)], 0.0)[0]["score"] == 0.3


def test_threshold_does_not_mutate_input():
    hits = [hit(0, 30, 0.1)]
    apply_threshold(hits, 0.25)
    assert hits == [hit(0, 30, 0.1)]


# ---- dedupe
def test_overlapping_neighbours_keep_the_higher_score():
    a, b = hit(0, 30, 0.6), hit(26, 56, 0.8)  # 4 s overlap, like real chunks
    assert dedupe_overlapping([a, b]) == [b]
    assert dedupe_overlapping([b, a]) == [b]  # input order is irrelevant


@pytest.mark.parametrize("overlap", [4, 7.5, 11])
def test_real_world_overlap_sizes(overlap):
    first, second = hit(0, 30, 0.7), hit(30 - overlap, 60 - overlap, 0.5)
    assert dedupe_overlapping([first, second]) == [first]


def test_chain_where_middle_is_best_drops_both_neighbours():
    a, b, c = hit(0, 30, 0.5), hit(25, 55, 0.9), hit(50, 80, 0.6)
    assert dedupe_overlapping([a, b, c]) == [b]


def test_chain_where_end_is_best_keeps_non_overlapping_third():
    a, b, c = hit(0, 30, 0.9), hit(25, 55, 0.8), hit(50, 80, 0.7)  # c does not overlap a
    assert spans(dedupe_overlapping([a, b, c])) == [("v1", 0, 30), ("v1", 50, 80)]


def test_long_chain_greedy_by_score():
    hits = [hit(i * 25, i * 25 + 30, s) for i, s in enumerate([0.5, 0.6, 0.95, 0.4, 0.7, 0.3])]
    out = dedupe_overlapping(hits)
    assert [h["score"] for h in out] == [0.95, 0.7, 0.5]  # no two survivors overlap
    for i, x in enumerate(out):
        for y in out[i + 1:]:
            assert x["end_sec"] <= y["start_sec"] or y["end_sec"] <= x["start_sec"]


def test_touching_ranges_are_not_overlapping():
    a, b = hit(0, 30, 0.8), hit(30, 60, 0.7)
    assert dedupe_overlapping([a, b]) == [a, b]


def test_same_times_in_different_videos_are_both_kept():
    a, b = hit(0, 30, 0.8, "v1"), hit(0, 30, 0.7, "v2")
    assert dedupe_overlapping([a, b]) == [a, b]


def test_contained_range_is_an_overlap():
    big, small = hit(0, 60, 0.6), hit(10, 20, 0.9)
    assert dedupe_overlapping([big, small]) == [small]


def test_identical_scores_prefer_earlier_start_deterministically():
    a, b = hit(0, 30, 0.7), hit(25, 55, 0.7)
    assert dedupe_overlapping([b, a]) == [a]
    assert dedupe_overlapping([a, b]) == [a]


def test_output_is_sorted_best_first_and_input_untouched():
    hits = [hit(100, 130, 0.4), hit(0, 30, 0.9), hit(200, 230, 0.6)]
    snapshot = copy.deepcopy(hits)
    assert [h["score"] for h in dedupe_overlapping(hits)] == [0.9, 0.6, 0.4]
    assert hits == snapshot


def test_dedupe_empty_and_single():
    assert dedupe_overlapping([]) == []
    assert dedupe_overlapping([hit(0, 1, 0.5)]) == [hit(0, 1, 0.5)]


def test_threshold_then_dedupe_pipeline():
    hits = [hit(0, 30, 0.9), hit(25, 55, 0.8), hit(200, 230, 0.2), hit(300, 330, 0.5)]
    assert spans(dedupe_overlapping(apply_threshold(hits, 0.25))) == [("v1", 0, 30), ("v1", 300, 330)]


# ---- sentences / highlight helpers
def test_split_sentences_on_punctuation():
    assert split_sentences("Hello there. How are you? Fine!  Thanks") == ["Hello there.", "How are you?", "Fine!", "Thanks"]


def test_split_sentences_ignores_blank_and_handles_no_punctuation():
    assert split_sentences("   ") == []
    assert split_sentences("no punctuation here") == ["no punctuation here"]


def test_long_run_on_without_punctuation_is_cut_by_word_count_and_loses_nothing():
    words = " ".join(f"w{i}" for i in range(100))
    pieces = split_sentences(words)
    assert len(pieces) == 3 and [len(p.split()) for p in pieces] == [40, 40, 20]
    assert " ".join(pieces) == words


def test_best_index_picks_highest_cosine_and_first_on_ties():
    q = [1.0, 0.0]
    assert best_index(q, [[0.0, 1.0], [0.8, 0.6], [0.6, 0.8]]) == 1
    assert best_index(q, [[0.8, 0.6], [0.8, 0.6]]) == 0
    assert best_index(np.array(q), np.array([[0.6, 0.8]])) == 0


# ---- highlight sentences: complete sentences, no dangling ends
# Real chunk 0.0-28.0 s of video 8f27bfee-4387-448e-bc0f-29cb05d75f80. Its last sentence is 51 words; the old splitter
# cut it into 20-word windows and the highlight became "world, I've discovered ... how they use their voice, how".
REAL_CHUNK_0 = (
    "Every time you open your mouth and you speak, people unconsciously categorize you into one of these three "
    "categories. Are you an awkward communicator? Are you a pretty good communicator? Or are you a natural "
    "communicator? And after 15 years as a communication coach, training Fortune 500 companies and speaking to "
    "millions of people around the world, I've discovered that you can tell which level someone sits at based on how "
    "they use their voice, how they use their body language, and how they use their words.")
REAL_CHUNK_56 = (
    "And yet we have this one second, two second, one minute interaction. Our brain goes into what you see as all "
    "there is. And you forget that there's so much else. What is the Latin theory? So it's a simple truth about "
    "life. Okay. If you want more peace, if you want more power, if you want more time and energy, stop trying to "
    "control and change other people, and learn how to let them be who they are, let them have their opinions, let "
    "them do what they're going to do, and then focus that time and energy back on yourself.")
TERMINATED = re.compile(r"[.!?][\"')\]]*$")


def test_real_chunk_splits_into_whole_sentences():
    s = split_sentences(REAL_CHUNK_0)
    assert s == [
        "Every time you open your mouth and you speak, people unconsciously categorize you into one of these three "
        "categories.",
        "Are you an awkward communicator?",
        "Are you a pretty good communicator?",
        "Or are you a natural communicator?",
        "And after 15 years as a communication coach, training Fortune 500 companies and speaking to millions of "
        "people around the world, I've discovered that you can tell which level someone sits at based on how they "
        "use their voice, how they use their body language, and how they use their words."]


def test_real_chunk_highlight_candidates_are_complete_sentences():
    for text in (REAL_CHUNK_0, REAL_CHUNK_56):
        for sentence in split_sentences(text):
            assert sentence[0].isupper(), sentence  # starts like a sentence
            assert TERMINATED.search(sentence), sentence  # ends like a sentence (so "...all there is." is fine)


def test_real_chunk_56_long_sentence_stays_in_one_piece():
    s = split_sentences(REAL_CHUNK_56)
    assert s[-1].startswith("If you want more peace,") and s[-1].endswith("back on yourself.")
    assert "Okay." in s and len(s) == 7


def test_the_old_broken_highlight_can_no_longer_be_produced():
    s = split_sentences(REAL_CHUNK_0)
    assert not any(x.startswith("world,") or x.endswith("how") or x.endswith("around the") for x in s)


def test_sentence_up_to_max_words_is_never_cut_but_run_ons_are():
    ok = " ".join(["word"] * 79) + " end."
    assert split_sentences(ok) == [ok]
    run_on = "So " + ", ".join(" ".join(["thing"] * 9) for _ in range(20)) + "."  # ~200 words, commas every 10
    pieces = split_sentences(run_on)
    assert len(pieces) > 1 and all(p.rstrip().endswith((",", ".")) for p in pieces)  # cut at clause boundaries
    assert all(len(p.split()) <= 50 for p in pieces)


def test_cut_never_lands_after_a_dangling_word():
    words = [f"w{i}" for i in range(100)]
    words[39] = "the"  # a plain 40-word cut would end the first piece on "the"
    words[79] = "and"
    pieces = split_sentences(" ".join(words))
    assert " ".join(pieces) == " ".join(words)  # moved forward, not dropped
    assert all(_bare(p.split()[-1]) not in _DANGLING for p in pieces)
    assert pieces[1].startswith("the w40")


def test_abbreviations_and_initials_do_not_split():
    assert split_sentences("Dr. Andrew Huberman said hello. Then he left.") == [
        "Dr. Andrew Huberman said hello.", "Then he left."]
    assert split_sentences("J. K. Rowling wrote it. Mr. Smith read it.") == [
        "J. K. Rowling wrote it.", "Mr. Smith read it."]
    assert split_sentences("Use tools, e.g. a hammer. Or i.e. anything heavy.") == [
        "Use tools, e.g. a hammer.", "Or i.e. anything heavy."]


def test_lowercase_after_a_period_is_not_a_boundary_but_numbers_and_quotes_are():
    assert split_sentences("Hmm... okay then. Fine.") == ["Hmm... okay then.", "Fine."]
    assert split_sentences("It is 5.5 feet tall. 3 people agreed.") == ["It is 5.5 feet tall.", "3 people agreed."]
    assert split_sentences('He said "stop." "Why?" she asked.') == ['He said "stop."', '"Why?" she asked.']
    assert split_sentences("So do I. Then we left.") == ["So do I.", "Then we left."]  # "I." is a word, not an initial


def test_fragment_at_a_chunk_edge_does_not_end_on_a_dangling_word():
    assert split_sentences("We talked about the importance of the") == ["We talked about the importance"]
    assert split_sentences("Great. And then we went to the") == ["Great.", "And then we went"]
    assert split_sentences("Well. Going to the") == ["Well.", "Going to the"]  # too short to trim
    assert split_sentences("That is what it is.") == ["That is what it is."]  # a finished sentence is left alone


@pytest.mark.parametrize("seed", range(40))
def test_random_text_invariants(seed):
    rng = random.Random(seed)
    vocab = ["the", "and", "to", "how", "speak", "voice", "Dr.", "e.g.", "people", "Fear", "2024", "of", "listen"]
    parts = []
    for _ in range(rng.randint(1, 60)):
        w = rng.choice(vocab)
        parts.append(w + rng.choice(["", "", "", ",", ".", "?", "!", "..."]))
    text = " ".join(parts)
    out = split_sentences(text)
    assert all(s and s == s.strip() for s in out)
    for s in out:
        if not TERMINATED.search(s) and not s.endswith((",", ";", ":")) and len(s.split()) > 3:
            assert _bare(s.split()[-1]) not in _DANGLING, s  # never ends on a dangling word
    assert [len(s.split()) for s in out].count(0) == 0
