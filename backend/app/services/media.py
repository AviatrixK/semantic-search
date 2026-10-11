import io
import json
import os
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


# ---- faststart: put the MP4 "moov" index at the start of the file so a browser can play without first reading the end.
FASTSTART_TYPES = {"mp4": "video/mp4", "m4v": "video/mp4", "mov": "video/quicktime"}  # webm/mkv have no moov atom


def _walk_boxes(f, size_total: int) -> bool | None:
    pos = 0
    while pos + 8 <= size_total:
        f.seek(pos)
        header = f.read(8)
        size, kind = int.from_bytes(header[:4], "big"), header[4:8]
        if size == 1:  # 64-bit size follows
            size = int.from_bytes(f.read(8), "big")
        elif size == 0:  # the box runs to the end of the file
            size = size_total - pos
        if kind == b"moov":
            return True
        if kind == b"mdat":
            return False
        if size < 8:
            return None
        pos += size
    return None


def moov_before_mdat(path: str) -> bool | None:
    """Walks the top-level MP4/MOV boxes (reads only their 8-16 byte headers). True: the index comes before the media data
    (already faststart). False: the media data comes first, so players must seek to the end for the index. None: neither
    box was found (not an MP4-family file)."""
    with open(path, "rb") as f:
        return _walk_boxes(f, os.path.getsize(path))


def moov_before_mdat_in_head(head: bytes) -> bool | None:
    """Same walk over only the first bytes of a file (a ranged S3 read). The boxes that matter sit at the very start, so
    True is reliable: that file is already faststart and need not be downloaded. Anything else (False, None) only means
    "not decided here": the caller must look at the whole file."""
    return _walk_boxes(io.BytesIO(head), len(head))


def needs_faststart(path: str) -> bool:
    """True only for an MP4-family file whose media data precedes its index."""
    ext = path.rsplit(".", 1)[-1].lower()
    return ext in FASTSTART_TYPES and moov_before_mdat(path) is False


def remux_faststart(src: str, dst: str) -> str:
    """Copies the streams (no re-encode, so no quality loss and seconds of work) into a new file whose index is first.
    Keeps every video and audio stream; subtitle and data tracks are not carried over. Raises MediaError if ffmpeg fails
    or the result is not actually faststart."""
    _require("ffmpeg")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", "0:v", "-map", "0:a?", "-c", "copy",
          "-movflags", "+faststart", dst])
    if moov_before_mdat(dst) is not True:
        raise MediaError("ffmpeg produced a file whose index is not at the start")
    return dst
