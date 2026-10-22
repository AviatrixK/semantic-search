"""The language list behind the upload dropdown: one source of truth for validation, Sarvam's code and Whisper's code."""
import re
from pathlib import Path

import pytest

from app.services import languages
from app.services.languages import CODES, LANGUAGES, normalize_language, sarvam_code, whisper_code

FRONTEND_LIST = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "languages.ts"


def test_it_is_the_23_languages_sarvam_documents():
    assert len(LANGUAGES) == len(CODES) == 23
    assert {"hi-IN", "en-IN", "ta-IN", "doi-IN", "sat-IN", "kok-IN"} <= CODES
    assert all(re.fullmatch(r"[a-z]{2,3}-IN", c) for c in CODES)


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO", "unknown"])
def test_no_choice_means_detect_automatically(value):
    assert normalize_language(value) is None
    assert sarvam_code(normalize_language(value)) == "unknown"  # Sarvam's own spelling of auto-detect
    assert whisper_code(normalize_language(value)) is None


def test_known_codes_normalise_and_unknown_ones_are_refused():
    assert normalize_language(" HI-in ") == "hi-IN"
    assert normalize_language("ta-IN") == "ta-IN"
    for bad in ("klingon", "hi", "xx-IN", "en-US", "<script>"):
        with pytest.raises(ValueError):
            normalize_language(bad)


def test_whisper_gets_its_own_codes_and_none_for_languages_it_does_not_know():
    assert (whisper_code("hi-IN"), whisper_code("en-IN"), whisper_code("od-IN")) == ("hi", "en", "or")  # Odia is "or" in Whisper
    for unsupported in ("kok-IN", "ks-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN"):
        assert whisper_code(unsupported) is None  # left to Whisper's detection rather than sending a code it would reject


def test_the_frontend_dropdown_offers_exactly_the_backend_list():
    """frontend/src/lib/languages.ts is a copy for the browser; this keeps the two from drifting apart."""
    text = FRONTEND_LIST.read_text(encoding="utf-8")
    front = re.findall(r"\{\s*code:\s*'([a-z]+-IN)',\s*label:\s*'([^']+)'\s*\}", text)
    assert front == list(LANGUAGES)
