import json
import subprocess
from unittest.mock import patch

import pytest

from app.services import media


def done(stdout=""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_missing_ffmpeg_raises_clear_error():
    with patch("app.services.media.shutil.which", return_value=None):
        with pytest.raises(media.ToolNotFoundError, match="ffmpeg"):
            media.extract_audio("in.mp4", "out.wav")
        with pytest.raises(media.ToolNotFoundError, match="ffprobe"):
            media.duration_sec("in.mp4")


def test_no_audio_stream_raises():
    with patch("app.services.media.shutil.which", return_value="/bin/x"), \
         patch("app.services.media.subprocess.run", return_value=done(json.dumps({"streams": []}))) as run:
        with pytest.raises(media.NoAudioStreamError):
            media.extract_audio("in.mp4", "out.wav")
        assert run.call_count == 1  # ffprobe only; ffmpeg never ran


def test_extract_audio_runs_ffprobe_then_ffmpeg():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        return done(json.dumps({"streams": [{"index": 1}]}))

    with patch("app.services.media.shutil.which", return_value="/bin/x"), \
         patch("app.services.media.subprocess.run", side_effect=fake_run):
        assert media.extract_audio("in.mp4", "out.wav") == "out.wav"
    assert calls == ["ffprobe", "ffmpeg"]


def test_ffmpeg_failure_wrapped():
    err = subprocess.CalledProcessError(1, ["ffprobe"], stderr="Invalid data found")
    with patch("app.services.media.shutil.which", return_value="/bin/x"), \
         patch("app.services.media.subprocess.run", side_effect=err):
        with pytest.raises(media.MediaError, match="Invalid data"):
            media.duration_sec("bad.mp4")


def test_duration_parsed():
    with patch("app.services.media.shutil.which", return_value="/bin/x"), \
         patch("app.services.media.subprocess.run", return_value=done(json.dumps({"format": {"duration": "12.9"}}))):
        assert media.duration_sec("in.mp4") == 12
