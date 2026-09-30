from functools import lru_cache

from app.core.config import settings


@lru_cache
def _model():
    from faster_whisper import WhisperModel  # heavy import: load once per process
    return WhisperModel(settings.WHISPER_SIZE, device="auto", compute_type="int8")


def transcribe(wav_path: str) -> tuple[list[dict], str]:
    """Returns ([{start, end, text}], detected_language)."""
    segments, info = _model().transcribe(wav_path, vad_filter=True)
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    return [s for s in out if s["text"]], info.language
