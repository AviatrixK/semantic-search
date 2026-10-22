"""List prices of the hosted providers, for ESTIMATES only (the eval's cost columns and the reprocess script's dry run). Billing is
whatever the provider charges: check the source links, prices change.

Embeddings are billed per input token; Sarvam speech-to-text per second of audio (INR). The prices below were read from the
providers' own pages on 2026-10-18. Where only a third party or an older announcement had a number, `confirmed` is False."""
from dataclasses import dataclass

AS_OF = "2026-10-18"


@dataclass(frozen=True)
class TokenPrice:
    usd_per_million_tokens: float
    source: str
    confirmed: bool = True  # False: not on the provider's current pricing page; see `note`
    note: str = ""


# key = "<provider>/<model>" (the "@<dimensions>" part of an embedder key is ignored: dimensions do not change the price)
EMBEDDING_PRICES: dict[str, TokenPrice] = {
    "gemini/gemini-embedding-2": TokenPrice(0.20, "https://ai.google.dev/gemini-api/docs/pricing", note="batch API $0.10; free tier exists"),
    "gemini/gemini-embedding-001": TokenPrice(0.15, "https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/", False,
                                              "from the 2025 launch post; not on the current pricing page"),
    "cohere/embed-v5.0-pro": TokenPrice(0.12, "https://cohere.com/blog/embed-5"),
    "cohere/embed-v5.0-fast": TokenPrice(0.08, "https://cohere.com/blog/embed-5"),
    "openai/text-embedding-3-large": TokenPrice(0.13, "https://developers.openai.com/api/docs/models/text-embedding-3-large"),
    "openai/text-embedding-3-small": TokenPrice(0.02, "https://developers.openai.com/api/docs/models/text-embedding-3-large"),
}

SARVAM_STT_INR_PER_HOUR = 30.0  # speech to text, ONE job per video (billed per second, rounded up per request)
SARVAM_STT_DIARIZATION_INR_PER_HOUR = 45.0  # not used by this app
SARVAM_SOURCE = "https://docs.sarvam.ai/api/pricing"


def model_key(embedder_key: str) -> str:
    """"gemini/gemini-embedding-2@1536" -> "gemini/gemini-embedding-2"."""
    return embedder_key.split("@", 1)[0]


def embedding_price(embedder_key: str) -> TokenPrice | None:
    """The price of an embedder, free for the local model, None when unknown (the caller must say "unknown", not guess)."""
    key = model_key(embedder_key)
    if key.startswith("local/"):
        return TokenPrice(0.0, "runs on this machine", note="CPU time only")
    return EMBEDDING_PRICES.get(key)


def embedding_cost_usd(tokens: float, price: TokenPrice) -> float:
    return tokens / 1_000_000 * price.usd_per_million_tokens


SARVAM_TEXT_INR_PER_10K_CHARS = 20.0  # /translate (sarvam-translate:v1 and mayura:v1) and /transliterate, per 10,000 characters of input
# Characters of chunk text per hour of video, MEASURED on the dev library (5 videos, 2.17 h, English speech: 169,177 characters of chunk
# text, which includes the 5 s overlap between chunks, so about 1.2x the transcript). Indic scripts and faster speakers will differ:
# pass real numbers (eval Corpus, `text_chars=`) when you have them.
TEXT_CHARS_PER_AUDIO_HOUR = 78_000


def sarvam_text_cost_inr(text_chars: float, variants: int = 2) -> float:
    """The text APIs bill every character sent: once for the romanisation, once for the English translation."""
    return text_chars * variants / 10_000 * SARVAM_TEXT_INR_PER_10K_CHARS


def sarvam_cost_inr(audio_seconds: float, variants: int = 2, text_chars: float | None = None) -> float:
    """Transcription of one video: ONE speech-to-text job (INR 30/h of audio) plus the text APIs for `variants` extra versions
    (romanised, English) of the chunk text. `text_chars` defaults to the measured TEXT_CHARS_PER_AUDIO_HOUR."""
    hours = audio_seconds / 3600
    chars = hours * TEXT_CHARS_PER_AUDIO_HOUR if text_chars is None else text_chars
    return hours * SARVAM_STT_INR_PER_HOUR + sarvam_text_cost_inr(chars, variants)
