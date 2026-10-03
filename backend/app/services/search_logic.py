"""Pure functions behind /api/search: no DB, no Redis, no models. Hits are dicts with at least
video_id, start_sec, end_sec and score (cosine similarity)."""
import re

import numpy as np

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def apply_threshold(hits: list[dict], min_score: float) -> list[dict]:
    """Drop hits scoring below `min_score` (a hit exactly at the threshold is kept)."""
    return [h for h in hits if h["score"] >= min_score]


def _overlaps(a: dict, b: dict) -> bool:
    """True if the two time ranges share any time. Ranges that merely touch (a.end == b.start) do not overlap."""
    return a["video_id"] == b["video_id"] and a["start_sec"] < b["end_sec"] and b["start_sec"] < a["end_sec"]


def dedupe_overlapping(hits: list[dict]) -> list[dict]:
    """Greedy, best score first: keep a hit unless it overlaps in time with a better hit that was already kept
    (same video only). Neighbouring chunks overlap by several seconds, so they often match the same query; this
    keeps the best one. Works for chains too: with A(0-30) B(25-55) C(50-80), if B scores highest only B survives;
    if A is best, A and C survive (C does not overlap A). Returns hits ordered by score, best first."""
    kept: list[dict] = []
    for hit in sorted(hits, key=lambda h: (-h["score"], h["video_id"], h["start_sec"])):
        if not any(_overlaps(hit, k) for k in kept):
            kept.append(hit)
    return kept


# --- sentence splitting (used for highlights) -------------------------------------------------------------
_TERMINATOR = re.compile(r"[.!?]+[\"')\]]*\s+")
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]*$")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")
_CLAUSE_END = re.compile(r"[,;:.!?][\"')\]]*$")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "u.s", "u.k", "fig",
                  "approx", "inc", "ltd"}
# Words a phrase cannot sensibly end on: cutting after one of them leaves a dangling fragment ("... around the").
_DANGLING = frozenset("""a an the of to in on at by for with from into onto about as than like over under between through
during before after and or but so nor yet because if when while that which who whom whose where whether although though
how why what whenever my your his her its our their this these those some any each every is are was were am be been being
do does did have has had will would can could should may might must not very just also then""".split())


def _bare(word: str) -> str:
    return word.strip(",;:.!?\"'()[]").lower()


def _sentences(text: str) -> list[str]:
    """Split on . ! ? followed by a capital letter, digit or opening quote. Not after abbreviations ("Dr.", "e.g.",
    initials like "J.") and not before lowercase text ("Hmm... okay"), which are not sentence ends."""
    out, start = [], 0
    for m in _TERMINATOR.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if not (nxt.isupper() or nxt.isdigit() or nxt in "\"'(["):
            continue
        words = text[:m.start()].split()
        word = _bare(words[-1]) if words else ""
        if m.group()[0] == "." and (word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha() and word != "i")):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def _trim_dangling(s: str) -> str:
    """A fragment (no end punctuation, e.g. cut off at a chunk edge) must not end on a dangling word."""
    if _ENDS_SENTENCE.search(s):
        return s
    words = s.split()
    while len(words) > 3 and _bare(words[-1]) in _DANGLING:
        words.pop()
    return " ".join(words)


def _split_long(sentence: str, target_words: int) -> list[str]:
    """Only for run-ons far longer than a real sentence (typically unpunctuated Whisper output). Cuts at clause
    punctuation (, ; :) where possible, packing clauses up to ~target_words; a single huge clause is cut by word count.
    A cut never lands after a dangling word: those words move to the start of the next piece."""
    pieces: list[str] = []
    cur: list[str] = []
    n = 0
    for clause in _CLAUSE_BREAK.split(sentence):
        words = clause.split()
        if len(words) > target_words:
            if cur:
                pieces.append(" ".join(cur))
                cur, n = [], 0
            pieces.extend(" ".join(words[i:i + target_words]) for i in range(0, len(words), target_words))
            continue
        if cur and n + len(words) > target_words:
            pieces.append(" ".join(cur))
            cur, n = [], 0
        cur.append(clause)
        n += len(words)
    if cur:
        pieces.append(" ".join(cur))
    for i in range(len(pieces) - 1):  # move dangling tail words forward (comma/period-ended pieces are fine as they are)
        if _CLAUSE_END.search(pieces[i]):
            continue
        words = pieces[i].split()
        moved = []
        while len(words) > 3 and _bare(words[-1]) in _DANGLING:
            moved.insert(0, words.pop())
        if moved:
            pieces[i] = " ".join(words)
            pieces[i + 1] = " ".join(moved + [pieces[i + 1]])
    if len(pieces) > 1 and len(pieces[-1].split()) < 5:  # no tiny trailing scrap
        pieces[-2] = f"{pieces[-2]} {pieces.pop()}"
    return pieces


def split_sentences(text: str, max_words: int = 80, target_words: int = 40) -> list[str]:
    """Split transcript text into sentences for highlights. A sentence is kept whole up to `max_words` (a real sentence
    is a better highlight than a clipped one); only longer run-ons are cut, at clause boundaries. Sentences start at a
    capital letter where the text allows it, and a fragment cut off at a chunk edge never ends on a dangling word."""
    text = " ".join(text.split())
    out: list[str] = []
    for sentence in _sentences(text):
        if len(sentence.split()) <= max_words:
            out.append(_trim_dangling(sentence))
        else:
            out.extend(_trim_dangling(p) for p in _split_long(sentence, target_words))
    return [s for s in out if s]


def best_index(query_vec, sentence_vecs) -> int:
    """Index of the sentence closest to the query. Vectors are L2-normalised (the embedder guarantees it), so the
    dot product is the cosine similarity. Ties go to the earliest sentence."""
    return int(np.argmax(np.asarray(sentence_vecs, dtype=np.float32) @ np.asarray(query_vec, dtype=np.float32)))
