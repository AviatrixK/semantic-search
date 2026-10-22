"""Languages an admin can pick on upload. The codes are Sarvam's (BCP-47 style, "hi-IN"); the same list drives the upload
dropdown (frontend/src/lib/languages.ts mirrors it: tests/test_languages.py keeps the two equal).

Source: https://docs.sarvam.ai/api/getting-started/models/saaras (23 languages: 22 Indian languages plus Indian English).
No language chosen = automatic detection (Sarvam `language_code=unknown`, Whisper's own detection)."""

LANGUAGES: tuple[tuple[str, str], ...] = (
    ("hi-IN", "Hindi"), ("en-IN", "English"), ("bn-IN", "Bengali"), ("ta-IN", "Tamil"), ("te-IN", "Telugu"),
    ("kn-IN", "Kannada"), ("ml-IN", "Malayalam"), ("mr-IN", "Marathi"), ("gu-IN", "Gujarati"), ("pa-IN", "Punjabi"),
    ("od-IN", "Odia"), ("as-IN", "Assamese"), ("ur-IN", "Urdu"), ("ne-IN", "Nepali"), ("kok-IN", "Konkani"),
    ("ks-IN", "Kashmiri"), ("sd-IN", "Sindhi"), ("sa-IN", "Sanskrit"), ("sat-IN", "Santali"), ("mni-IN", "Manipuri"),
    ("brx-IN", "Bodo"), ("mai-IN", "Maithili"), ("doi-IN", "Dogri"),
)
CODES = frozenset(code for code, _ in LANGUAGES)

# Whisper's own language codes. Languages Whisper does not know (Konkani, Kashmiri, Santali, Manipuri, Bodo, Maithili, Dogri)
# are left to its automatic detection.
_WHISPER = {"hi-IN": "hi", "en-IN": "en", "bn-IN": "bn", "ta-IN": "ta", "te-IN": "te", "kn-IN": "kn", "ml-IN": "ml",
            "mr-IN": "mr", "gu-IN": "gu", "pa-IN": "pa", "od-IN": "or", "as-IN": "as", "ur-IN": "ur", "ne-IN": "ne",
            "sa-IN": "sa", "sd-IN": "sd"}


def normalize_language(value: str | None) -> str | None:
    """"" / None / "auto" -> None (detect automatically). A known code -> that code (case-insensitive). Anything else: ValueError."""
    if value is None:
        return None
    v = value.strip()
    if not v or v.lower() in ("auto", "unknown"):
        return None
    for code in CODES:
        if code.lower() == v.lower():
            return code
    raise ValueError(f"Unsupported language {value!r}")


def whisper_code(code: str | None) -> str | None:
    """The faster-whisper `language` argument for a chosen language (None = let Whisper detect)."""
    return _WHISPER.get(code) if code else None


def sarvam_code(code: str | None) -> str:
    """Sarvam's `language_code` parameter: the chosen code, or "unknown" for automatic detection."""
    return code or "unknown"


# Languages the SDK (sarvamai 0.1.37, SpeechToTextJobParameters.language_code) lists under "Additional Options (saaras:v3 only)". Sarvam's
# model page says v4 covers "22 Indic languages", so the two sources disagree; the SDK is the safer reading and v3 costs the same, so with
# SARVAM_MODEL=saaras:v4 these go to the fallback model up front (the 400/422 retry in SarvamClient.create_job still covers anything else).
V3_ONLY = frozenset({"as-IN", "ur-IN", "ne-IN", "kok-IN", "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN"})


# ---- Sarvam text APIs (POST /transliterate and POST /translate): the languages each accepts as SOURCE, from
# https://docs.sarvam.ai/api-reference/text/transliterate-text and .../translate-text (read 2026-10-20).
TRANSLIT_SOURCES = frozenset({"bn-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"})  # + en-IN, auto
MAYURA_SOURCES = TRANSLIT_SOURCES | {"en-IN"}  # mayura:v1 translate: these 11 (and "auto")
SARVAM_TRANSLATE_SOURCES = CODES  # sarvam-translate:v1: all 23 languages of the dropdown
