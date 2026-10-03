import json
import shutil
import subprocess


class MediaError(Exception):
    """Problem with the media tooling or the input file.

    `str(exc)` is for developers (CLI, logs) and may contain paths. `user_message` is the short, path-free text
    that is safe to show users and store in jobs.error. `raw` is the full untruncated tool output, for logs only."""
    user_message = "This file is not a valid video."

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw if raw is not None else message


class ToolNotFoundError(MediaError):
    user_message = "Video processing is unavailable on the server. Please contact an administrator."


class NoAudioStreamError(MediaError):
    user_message = "This video has no audio track."


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
        raw = f"{e.stderr or ''}{e.stdout or ''}".strip()
        raise MediaError(f"{cmd[0]} failed (exit {e.returncode}): {raw[-500:]}", raw=raw) from e


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
    try:
        return int(float(json.loads(out)["format"]["duration"]))
    except (KeyError, ValueError, TypeError) as e:  # ffprobe succeeded but found no usable duration
        raise MediaError(f"ffprobe returned no duration: {out[:200]}", raw=out) from e
