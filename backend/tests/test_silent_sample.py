"""Real-file check for the "no speech" path using sample/EduSphereDemonstration.mp4 (audio track is silent,
-91 dB). Needs ffmpeg + ffprobe on PATH and the (gitignored) sample; skipped otherwise. Whisper is never loaded."""
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.services import media, transcription

SAMPLE = Path(__file__).resolve().parents[2] / "sample" / "EduSphereDemonstration.mp4"

pytestmark = pytest.mark.skipif(
    not (SAMPLE.exists() and shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="needs ffmpeg/ffprobe on PATH and sample/EduSphereDemonstration.mp4")


def test_sample_audio_is_silent_and_yields_no_speech(tmp_path):
    wav = media.extract_audio(str(SAMPLE), str(tmp_path / "audio.wav"))
    audio = transcription._load_wav(wav)
    assert audio.size > 0 and float(np.abs(audio).max()) < 0.01  # real ffmpeg output loads and is silent

    class SilentModel:  # what Whisper's VAD returns for silence: zero segments
        def transcribe(self, audio, **kw):
            return iter([]), type("Info", (), {"language": "en"})()

    with patch.object(transcription, "_model", return_value=SilentModel()):
        with pytest.raises(transcription.NoSpeechError):
            transcription.transcribe(wav)
