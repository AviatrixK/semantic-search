import wave
from functools import lru_cache
from typing import Callable

import numpy as np

from app.core.config import settings

SAMPLE_RATE = 16000


class NoSpeechError(Exception):
    """Transcription produced no text (silent or music-only audio)."""


@lru_cache
def _model():
    from faster_whisper import WhisperModel  # heavy import: load once per process
    return WhisperModel(settings.WHISPER_SIZE, device="auto", compute_type="int8")


def _load_wav(path: str) -> np.ndarray:
    """Read the 16 kHz mono 16-bit WAV that ffmpeg produced, as float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("Expected 16 kHz mono 16-bit WAV from extract_audio()")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav_path: str, on_progress: Callable[[float, float], None] | None = None) -> tuple[list[dict], str]:
    """Returns ([{start, end, text}], detected_language). Always hands faster-whisper a numpy array, never a path.
    `on_progress(done_sec, total_sec)` is called after each segment (segments are a lazy generator, so this is
    where the transcription time is actually spent)."""
    audio = _load_wav(wav_path)
    total = len(audio) / SAMPLE_RATE
    segments, info = _model().transcribe(audio, vad_filter=True)
    out = []
    for s in segments:
        out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        if on_progress:
            on_progress(s.end, total)
    out = [s for s in out if s["text"]]
    if not out:
        raise NoSpeechError("No speech detected in this video")
    return out, info.language
