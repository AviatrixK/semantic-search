"""Download the output of an EXISTING Sarvam batch job so its real shape can be inspected. Starts no job and uploads nothing.

  docker compose exec api python -m app.scripts.sarvam_debug --job-id 20261022_93a382f8-cc47-41e0-9fc7-056ab4bae266

Calls only GET /speech-to-text/job/v1/{id}/status and POST /speech-to-text/job/v1/download-files, then fetches the output files from the
pre-signed URLs it hands back. Writes debug/sarvam_<job_id>.json (relative to the current directory; `debug/` is gitignored) holding the
job state, the file names and the raw output JSON. Nothing secret is saved: no API key, no request headers and no pre-signed URLs (their
query string is a credential). It also prints the top-level keys of every output and where anything that looks like timestamps is."""
import argparse
import json
import sys
from pathlib import Path

from app.services.sarvam import SarvamClient, SarvamError


def find_timestamp_paths(obj, prefix: str = "") -> list[str]:
    """Paths (like timestamps.words[].start) of every key whose name suggests timing, anywhere in the output."""
    hints = ("time", "start", "end", "offset", "duration", "stamp")
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if any(h in str(k).lower() for h in hints):
                found.append(path)
            found += find_timestamp_paths(v, path)
    elif isinstance(obj, list) and obj:
        found += find_timestamp_paths(obj[0], f"{prefix}[]")  # the first element stands for the rest
    return list(dict.fromkeys(found))


def describe(obj, depth: int = 0, max_depth: int = 3) -> str:
    """A compact picture of the shape: keys and types, lists by length and the shape of their first element."""
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"{{...{len(obj)} keys}}"
        lines = [f"{pad}  {k}: {describe(v, depth + 1, max_depth)}" for k, v in obj.items()]
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(obj, list):
        first = describe(obj[0], depth + 1, max_depth) if obj else "-"
        return f"list[{len(obj)}] of {first}"
    if isinstance(obj, str):
        return f"str({len(obj)})"
    return f"{type(obj).__name__}({obj!r})"


def fetch(job_id: str, client: SarvamClient | None = None) -> dict:
    client = client or SarvamClient()
    status = client.status(job_id)
    names = []
    for d in status.get("job_details") or []:
        for out in d.get("outputs") or []:
            if out.get("file_name"):
                names.append(out["file_name"])
    result = {"job_id": job_id, "job_state": status.get("job_state"), "status_keys": sorted(status),
              "job_details": [{k: v for k, v in d.items() if k in ("state", "inputs", "outputs", "error_message", "exception_name")}
                              for d in status.get("job_details") or []], "outputs": {}}
    if names:
        for name, payload in client.download_outputs(job_id, names).items():
            result["outputs"][name] = payload
    return result


def main(argv: list[str] | None = None, *, client: SarvamClient | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--out-dir", default="debug", help="where to write sarvam_<job_id>.json (default: ./debug)")
    args = ap.parse_args(argv)
    if not all(c.isalnum() or c in "-_" for c in args.job_id):
        ap.error("job id may only contain letters, digits, - and _")
    try:
        result = fetch(args.job_id, client)
    except SarvamError as exc:
        print(f"Sarvam call failed: {type(exc).__name__}: {exc}")
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sarvam_{args.job_id}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"job {args.job_id}: state {result['job_state']}; status keys: {result['status_keys']}")
    if not result["outputs"]:
        print("no output files listed in job_details")
    for name, payload in result["outputs"].items():
        print(f"\noutput {name}:")
        if isinstance(payload, dict):
            print(f"  top-level keys: {list(payload)}")
            print(f"  shape: {describe(payload)}")
            print(f"  timing-looking paths: {find_timestamp_paths(payload) or 'NONE'}")
        else:
            print(f"  not an object: {type(payload).__name__}")
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
