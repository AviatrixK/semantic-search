"""Standalone transcription — no Docker, no database. Reuses the same services as the worker.

Setup:
    pip install faster-whisper pydantic-settings    # ffmpeg must also be installed (winget/brew/apt)
Run:
    python scripts/transcribe.py path/to/video.mp4 [--model base] [--chunks]
Output:
    prints timestamped segments and writes <video>.transcript.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import settings  # noqa: E402
from app.services import chunking, media, transcription  # noqa: E402


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default=None, help="tiny|base|small|medium (default: WHISPER_SIZE setting)")
    p.add_argument("--chunks", action="store_true", help="also print the chunk windows used for embedding")
    args = p.parse_args()

    if args.model:
        settings.WHISPER_SIZE = args.model

    video = Path(args.video)
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = media.extract_audio(str(video), str(Path(tmp) / "audio.wav"))
            segments, language = transcription.transcribe(wav)
    except media.MediaError as e:
        sys.exit(f"error: {e}")

    for s in segments:
        print(f"[{fmt(s['start'])} -> {fmt(s['end'])}] {s['text']}")

    if args.chunks:
        spans = chunking.window(segments)
        print(f"\n--- {len(spans)} chunks (size={settings.CHUNK_SECONDS}s, overlap={settings.CHUNK_OVERLAP}s) ---")
        for c in spans:
            print(f"[{fmt(c.start)} -> {fmt(c.end)}] {c.text[:80]}")

    dest = video.with_suffix(".transcript.json")
    dest.write_text(json.dumps({"language": language, "segments": segments}, indent=2), encoding="utf-8")
    print(f"\n{len(segments)} segments, language={language}, {time.time() - t0:.1f}s -> {dest}")


if __name__ == "__main__":
    main()
