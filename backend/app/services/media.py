import json
import shutil
import subprocess


class MediaError(Exception):
    """Base class for problems with the media tooling or the input file."""


class ToolNotFoundError(MediaError):
    pass


class NoAudioStreamError(MediaError):
    pass


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise ToolNotFoundError(f"'{tool}' was not found on PATH. Install ffmpeg (winget install ffmpeg / "
                                f"brew install ffmpeg / apt install ffmpeg) and restart your shell.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:  # tool vanished between the PATH check and the call
        raise ToolNotFoundError(f"'{cmd[0]}' could not be executed: {e}") from e
    except subprocess.CalledProcessError as e:
        raise MediaError(f"{cmd[0]} failed (exit {e.returncode}): {(e.stderr or '').strip()[-500:]}") from e


def has_audio(path: str) -> bool:
    _require("ffprobe")
    out = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                "-of", "json", path]).stdout
    return bool(json.loads(out or "{}").get("streams"))


def extract_audio(video_path: str, wav_path: str) -> str:
    """16 kHz mono WAV is what Whisper expects."""
    _require("ffmpeg")
    if not has_audio(video_path):
        raise NoAudioStreamError(f"{video_path} has no audio stream, nothing to transcribe")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path])
    return wav_path


def duration_sec(path: str) -> int:
    _require("ffprobe")
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]).stdout
    return int(float(json.loads(out)["format"]["duration"]))
