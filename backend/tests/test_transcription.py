import wave

import numpy as np
import pytest

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
