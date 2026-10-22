"""Audio helpers for hosted transcription: compressed MP3 extraction and splitting under the provider's length limit.
The command-line tests use mocks; the last group runs the real ffmpeg (skipped when it is not installed)."""
import json
import os
import shutil
import subprocess
from unittest.mock import patch

import pytest

from app.services import media

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")


def done(stdout=""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_mp3_extraction_asks_for_mono_16k_mp3_at_a_speech_bitrate():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return done(json.dumps({"streams": [{"index": 1}]}))

    with patch.object(media.shutil, "which", return_value="/bin/x"), patch.object(media.subprocess, "run", side_effect=fake_run):
        assert media.extract_audio_mp3("in.mp4", "out.mp3") == "out.mp3"
    ffmpeg = calls[-1]
    assert ffmpeg[0] == "ffmpeg" and "libmp3lame" in ffmpeg and ffmpeg[ffmpeg.index("-ac") + 1] == "1"
    assert ffmpeg[ffmpeg.index("-ar") + 1] == "16000" and ffmpeg[ffmpeg.index("-b:a") + 1] == "48k" and "-vn" in ffmpeg


def test_a_video_without_audio_is_refused_before_ffmpeg_runs():
    with patch.object(media.shutil, "which", return_value="/bin/x"), \
         patch.object(media.subprocess, "run", return_value=done(json.dumps({"streams": []}))) as run:
        with pytest.raises(media.NoAudioStreamError):
            media.extract_audio_mp3("in.mp4", "out.mp3")
    assert run.call_count == 1  # ffprobe only


def test_short_audio_is_not_split(tmp_path):
    with patch.object(media, "duration_float", return_value=3600.0), patch.object(media, "_run") as run:
        assert media.split_audio("a.mp3", str(tmp_path), 6600) == [("a.mp3", 0.0)]
    run.assert_not_called()


def test_the_exact_duration_keeps_the_fraction():
    out = json.dumps({"format": {"duration": "12.9"}})
    with patch.object(media.shutil, "which", return_value="/bin/x"), patch.object(media.subprocess, "run", return_value=done(out)):
        assert media.duration_float("a.mp3") == 12.9 and media.duration_sec("a.mp3") == 12


@needs_ffmpeg
def test_real_ffmpeg_makes_a_small_mono_mp3_from_a_video(tmp_path):
    video = str(tmp_path / "v.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=6:size=160x120:rate=10", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=6", "-shortest", "-c:v", "libx264", "-c:a", "aac", video], check=True)
    mp3 = media.extract_audio_mp3(video, str(tmp_path / "a.mp3"))
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels,sample_rate", "-of", "json", mp3],
                                     capture_output=True, text=True).stdout)["streams"][0]
    assert (info["codec_name"], info["channels"], info["sample_rate"]) == ("mp3", 1, "16000")
    assert 5.5 < media.duration_float(mp3) < 6.6
    assert os.path.getsize(mp3) < 0.5 * os.path.getsize(video) + 40_000


@needs_ffmpeg
def test_real_ffmpeg_splits_long_audio_and_the_offsets_rebuild_one_timeline(tmp_path):
    src = str(tmp_path / "long.mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=25", "-c:a", "libmp3lame",
                    "-b:a", "32k", src], check=True)
    out_dir = tmp_path / "parts"
    out_dir.mkdir()
    parts = media.split_audio(src, str(out_dir), 10)
    assert len(parts) >= 3  # 25 s in parts of about 10 s
    offsets = [o for _, o in parts]
    assert offsets[0] == 0.0 and offsets == sorted(offsets)
    total = media.duration_float(src)
    assert offsets[-1] + media.duration_float(parts[-1][0]) == pytest.approx(total, abs=0.3)  # parts + offsets cover the whole audio
    assert all(os.path.exists(p) for p, _ in parts)
