import wave
from unittest.mock import patch

import numpy as np
import pytest

from app.services import transcription
from app.services.transcription import _load_wav


def write_wav(path, rate=16000, channels=1, width=2, frames=b"\x00\x00\x00\x40\x00\xc0\xff\x7f"):
    """Default frames are the int16 samples [0, 16384, -16384, 32767]."""
    with wave.open(str(path), "wb") as w:
        w.setframerate(rate)
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.writeframes(frames)
    return str(path)


def test_load_wav_returns_float32_in_range(tmp_path):
    arr = _load_wav(write_wav(tmp_path / "ok.wav"))
    assert arr.dtype == np.float32 and arr.shape == (4,)
    assert np.allclose(arr, [0, 0.5, -0.5, 32767 / 32768])


def test_load_wav_empty_audio(tmp_path):
    assert _load_wav(write_wav(tmp_path / "empty.wav", frames=b"")).size == 0


@pytest.mark.parametrize("kw", [
    {"rate": 44100},
    {"rate": 8000},
    {"channels": 2},
    {"width": 1, "frames": b"\x00\x01"},
])
def test_load_wav_rejects_wrong_format(tmp_path, kw):
    with pytest.raises(ValueError):
        _load_wav(write_wav(tmp_path / "bad.wav", **kw))


class FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class FakeModel:
    def __init__(self, segs):
        self.segs, self.got = segs, None

    def transcribe(self, audio, **kw):
        self.got = audio
        return iter(self.segs), type("Info", (), {"language": "en"})()


def test_transcribe_passes_array_not_path_and_cleans_segments(tmp_path):
    model = FakeModel([FakeSeg(0.004, 1.236, " hello "), FakeSeg(1.3, 2.0, "   ")])
    with patch("app.services.transcription._model", return_value=model):
        segs, lang = transcription.transcribe(write_wav(tmp_path / "a.wav"))
    assert isinstance(model.got, np.ndarray) and model.got.dtype == np.float32
    assert segs == [{"start": 0.0, "end": 1.24, "text": "hello"}] and lang == "en"


@pytest.mark.parametrize("segs", [[], [FakeSeg(0, 1, "  "), FakeSeg(1, 2, "")]])
def test_no_speech_raises(tmp_path, segs):
    with patch("app.services.transcription._model", return_value=FakeModel(segs)):
        with pytest.raises(transcription.NoSpeechError, match="No speech detected in this video"):
            transcription.transcribe(write_wav(tmp_path / "a.wav"))


def test_progress_callback_reports_position_and_duration(tmp_path):
    model = FakeModel([FakeSeg(0, 1.0, "a"), FakeSeg(1.0, 2.0, "b")])
    seen = []
    one_sec = b"\x00\x00" * 16000
    with patch("app.services.transcription._model", return_value=model):
        transcription.transcribe(write_wav(tmp_path / "a.wav", frames=one_sec * 4),
                                 on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1.0, 4.0), (2.0, 4.0)]


def test_whisper_model_gets_the_configured_thread_count():
    import sys
    from types import SimpleNamespace
    seen = {}

    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    transcription._model.cache_clear()
    try:
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisper)}), \
             patch.object(transcription.settings, "WHISPER_THREADS", 1):
            transcription._model()
        assert seen["cpu_threads"] == 1
        transcription._model.cache_clear()
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisper)}), \
             patch.object(transcription.settings, "WHISPER_THREADS", 0):
            transcription._model()
        assert seen["cpu_threads"] == 1  # 0 would mean "all cores" to ctranslate2: never allowed
    finally:
        transcription._model.cache_clear()


def test_worker_defaults_leave_a_cpu_for_the_api():
    from app.core.config import Settings
    assert Settings.model_fields["WHISPER_THREADS"].default == 1
    assert Settings.model_fields["TORCH_THREADS"].default == 1
