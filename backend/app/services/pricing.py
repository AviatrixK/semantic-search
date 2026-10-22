"""List prices of the hosted providers, for ESTIMATES only (the eval's cost columns and the reprocess script's dry run). Billing is
whatever the provider charges: check the source links, prices change.

Embeddings are billed per input token; Sarvam speech-to-text per second of audio (INR). The prices below were read from the
providers' own pages on 2026-10-18. Where only a third party or an older announcement had a number, `confirmed` is False."""
import math
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


# ---- Gemini Flash-Lite text models (TEXT_VARIANT_MODEL): USD per 1M tokens, STANDARD (not batch) prices, text input. Output includes
# thinking tokens. Read on 2026-10-22 from https://ai.google.dev/gemini-api/docs/pricing
GEMINI_PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"


@dataclass(frozen=True)
class TextModelPrice:
    usd_in_per_million: float
    usd_out_per_million: float
    source: str = GEMINI_PRICING_SOURCE
    note: str = ""


TEXT_MODEL_PRICES: dict[str, TextModelPrice] = {
    "gemini-3.1-flash-lite": TextModelPrice(0.25, 1.50),
    "gemini-3.5-flash-lite": TextModelPrice(0.30, 2.50, note="Google's recommended Flash-Lite for new projects"),
    "gemini-2.5-flash-lite": TextModelPrice(0.10, 0.40, note="not deprecated, but access is limited to users who used it before"),
}

# ESTIMATE ASSUMPTIONS for the text-variant cost before any real run. NONE of these was measured: they are guesses about Gemini's
# tokeniser on Indic text. Once videos have been processed, videos.text_variants_in_tokens / _out_tokens hold the real numbers: replace
# these constants (or just pass measured tokens) when they differ.
INDIC_CHARS_PER_TOKEN = 2.5  # native-script input
ENGLISH_CHARS_PER_TOKEN = 4.0
ROMAN_CHARS_PER_TOKEN = 3.0  # Hinglish spelling is longer and less regular than English
ROMAN_LENGTH_FACTOR = 1.2  # characters of the romanised text relative to the native text
EN_LENGTH_FACTOR = 1.0
JSON_OVERHEAD = 1.1  # ids, keys and quotes around every answer
PROMPT_TOKENS_PER_REQUEST = 350  # the instructions sent with every request
CHUNKS_PER_AUDIO_HOUR = 175  # measured on the dev library: 380 chunks in 2.17 h
CHARS_PER_CHUNK = TEXT_CHARS_PER_AUDIO_HOUR / CHUNKS_PER_AUDIO_HOUR  # 446 characters, as measured


def estimate_text_variant_tokens(native_chars: float, chunks: float, batch_items: int = 15, batch_chars: int = 6000) -> tuple[int, int]:
    """(input tokens, output tokens) to make the English and romanised text of `chunks` chunks holding `native_chars` characters."""
    if native_chars <= 0 or chunks <= 0:
        return 0, 0
    per_chunk = native_chars / chunks
    per_request = max(1, min(batch_items, int(batch_chars // max(per_chunk, 1))))
    requests = math.ceil(chunks / per_request)
    tokens_in = native_chars / INDIC_CHARS_PER_TOKEN + requests * PROMPT_TOKENS_PER_REQUEST
    tokens_out = (native_chars * EN_LENGTH_FACTOR / ENGLISH_CHARS_PER_TOKEN + native_chars * ROMAN_LENGTH_FACTOR / ROMAN_CHARS_PER_TOKEN) * JSON_OVERHEAD
    return round(tokens_in), round(tokens_out)


def text_model_price(model: str) -> TextModelPrice | None:
    return TEXT_MODEL_PRICES.get(model.removeprefix("models/"))


def text_model_cost_usd(model: str, tokens_in: float, tokens_out: float) -> float | None:
    """None when the model's price is not in the table: an unknown price is reported as unknown, never guessed."""
    price = text_model_price(model)
    if price is None:
        return None
    return tokens_in / 1e6 * price.usd_in_per_million + tokens_out / 1e6 * price.usd_out_per_million


def text_variants_per_hour(provider: str, model: str | None = None, text_chars_per_hour: float | None = None,
                           inr_per_usd: float | None = None) -> dict:
    """What making text_en and text_roman costs per hour of video for a TEXT_VARIANT_PROVIDER: {"inr", "usd", "known", "tokens_in",
    "tokens_out"}. gemini is priced in USD (INR only when an exchange rate is supplied), sarvam in INR, none is free. `known` is False
    when the model's price is not in the table above (then the numbers are None, never a guess)."""
    chars = TEXT_CHARS_PER_AUDIO_HOUR if text_chars_per_hour is None else text_chars_per_hour
    out = {"provider": provider, "model": model, "inr": 0.0, "usd": 0.0, "known": True, "tokens_in": 0, "tokens_out": 0}
    if provider == "sarvam":
        out["inr"] = sarvam_text_cost_inr(chars, 2)
        out["usd"] = out["inr"] / inr_per_usd if inr_per_usd else None
    elif provider == "gemini":
        out["tokens_in"], out["tokens_out"] = estimate_text_variant_tokens(chars, chars / CHARS_PER_CHUNK)
        usd = text_model_cost_usd(model or "", out["tokens_in"], out["tokens_out"])
        out["known"] = usd is not None
        out["usd"] = usd
        out["inr"] = usd * inr_per_usd if usd is not None and inr_per_usd else None
    return out
