import json
import subprocess


def extract_audio(video_path: str, wav_path: str) -> str:
    """16 kHz mono WAV is what Whisper expects."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                    wav_path], check=True)
    return wav_path


def duration_sec(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return int(float(json.loads(out)["format"]["duration"]))
