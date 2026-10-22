"""Cost estimates for the provider comparison. Pure functions (no database, no network); the prices live in app/services/pricing.py.

Two numbers per embedding provider:
  per 1,000 searches   = average query tokens x 1000 x price. Queries are embedded once per search (a cache hit is free, so this is the
                         worst case: every query new).
  per hour of video    = tokens of ALL the text that gets embedded for one hour of video x price. Chunks overlap by 5 of 30 seconds, so
                         the text embedded is about 1.2x the transcript; every text variant that is embedded (native, English) counts.
And per hour of video for transcription: Sarvam INR 30/h for the ONE speech-to-text job, plus INR 20 per 10K characters for each
text version made by the text APIs (romanised, English) from the native chunk text.

Token counts are an ESTIMATE (about 4 characters per token) unless the provider reported its own usage; Indic scripts use more tokens
per character, so for Hindi and friends the real numbers are higher. Compare providers with that in mind and check the bill."""
from dataclasses import dataclass

from app.services import pricing
from app.services.embedders import estimate_tokens


@dataclass(frozen=True)
class Corpus:
    """The videos the eval runs on, as stored."""
    videos: int
    hours: float
    native_chars: int
    en_chars: int  # 0 for videos transcribed by Whisper (no English variant)
    chunks: int


def corpus_tokens(corpus: Corpus, *, assume_both_variants: bool = False) -> int:
    """Estimated tokens embedded for the whole corpus. `assume_both_variants`: pretend every chunk also has an English variant as long
    as its native text, which is what Sarvam transcription would add for videos that Whisper transcribed so far."""
    chars = corpus.native_chars + (corpus.en_chars if not assume_both_variants else max(corpus.en_chars, corpus.native_chars))
    return max(1, chars // 4) if chars else 0


def tokens_per_hour(corpus: Corpus, *, assume_both_variants: bool = False) -> float | None:
    return corpus_tokens(corpus, assume_both_variants=assume_both_variants) / corpus.hours if corpus.hours > 0 else None


def average_query_tokens(queries: list[str]) -> float:
    return sum(estimate_tokens(q) for q in queries) / len(queries) if queries else 0.0


def per_1000_searches_usd(avg_query_tokens: float, price: pricing.TokenPrice) -> float:
    return pricing.embedding_cost_usd(avg_query_tokens * 1000, price)


def per_video_hour_usd(corpus: Corpus, price: pricing.TokenPrice, *, assume_both_variants: bool = False) -> float | None:
    tph = tokens_per_hour(corpus, assume_both_variants=assume_both_variants)
    return pricing.embedding_cost_usd(tph, price) if tph is not None else None


def transcription_per_hour_inr(transcriber: str, variants: int = 2, text_chars_per_hour: float | None = None) -> float:
    """INR per hour of video: one speech-to-text job plus `variants` text-API versions of the chunk text. Whisper runs locally."""
    if transcriber != "sarvam":
        return 0.0
    return pricing.sarvam_cost_inr(3600, variants, text_chars_per_hour)


@dataclass(frozen=True)
class CostRow:
    embedder_key: str
    price: pricing.TokenPrice | None
    per_1000_searches_usd: float | None
    per_video_hour_usd: float | None


def cost_row(embedder_key: str, avg_query_tokens: float, corpus: Corpus, *, assume_both_variants: bool = False) -> CostRow:
    """Costs for one embedder; every number is None when its price is unknown (never guessed)."""
    price = pricing.embedding_price(embedder_key)
    if price is None:
        return CostRow(embedder_key, None, None, None)
    return CostRow(embedder_key, price, per_1000_searches_usd(avg_query_tokens, price),
                   per_video_hour_usd(corpus, price, assume_both_variants=assume_both_variants))


def fmt_usd(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "free" if value == 0 else f"${value:.4f}" if value < 1 else f"${value:.2f}"


def combined_per_hour_usd(row: CostRow, transcriber: str, variants: int, inr_per_usd: float | None) -> float | None:
    """Embedding + transcription per hour of video in USD, only when the caller supplied an exchange rate (none is assumed)."""
    if row.per_video_hour_usd is None or (transcriber == "sarvam" and not inr_per_usd):
        return None
    inr = transcription_per_hour_inr(transcriber, variants)
    return row.per_video_hour_usd + (inr / inr_per_usd if inr else 0.0)
