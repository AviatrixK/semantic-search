"""Week 1: standalone transcription — no Docker, no database.

Setup:
    pip install faster-whisper          # ffmpeg must also be installed (winget/brew/apt)
Run:
    python scripts/transcribe.py path/to/video.mp4 [--model base]
Output:
    prints timestamped segments and writes <video>.transcript.json
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="base", help="tiny|base|small|medium")
    args = p.parse_args()

    from faster_whisper import WhisperModel

    video = Path(args.video)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        str(wav)], check=True)
        t0 = time.time()
        model = WhisperModel(args.model, device="auto", compute_type="int8")
        segments, info = model.transcribe(str(wav), vad_filter=True)
        out = []
        for s in segments:  # generator: transcription happens while iterating
            print(f"[{fmt(s.start)} -> {fmt(s.end)}] {s.text.strip()}")
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})

    dest = video.with_suffix(".transcript.json")
    dest.write_text(json.dumps({"language": info.language, "segments": out}, indent=2))
    print(f"\n{len(out)} segments, language={info.language}, {time.time() - t0:.1f}s -> {dest}")


if __name__ == "__main__":
    main()
