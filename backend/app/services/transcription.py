import wave

import numpy as np
from functools import lru_cache

from app.core.config import settings


class NoSpeechError(Exception):
    """Transcription produced no text (silent or music-only audio)."""


@lru_cache
def _model():
    from faster_whisper import WhisperModel  # heavy import: load once per process
    return WhisperModel(settings.WHISPER_SIZE, device="auto", compute_type="int8")

def _load_wav(path: str) -> np.ndarray:
    """Read the 16 kHz mono 16-bit WAV that ffmpeg produced, as float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected 16 kHz mono 16-bit WAV from extract_audio()")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

def transcribe(wav_path: str) -> tuple[list[dict], str]:
    """Returns ([{start, end, text}], detected_language)."""
    segments, info = _model().transcribe(_load_wav(wav_path), vad_filter=True)
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    out = [s for s in out if s["text"]]
    if not out:
        raise NoSpeechError("No speech detected in this video")
    return out, info.language
