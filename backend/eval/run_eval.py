"""Retrieval evaluation: for each search mode, how often does the correct moment come back, and how fast?

    docker compose exec api python eval/run_eval.py                  # all four modes, eval/queries.jsonl
    docker compose exec api python eval/run_eval.py --skip-rerank    # no cross-encoder download
    docker compose exec api python eval/run_eval.py --queries eval/other.jsonl --k 10 --repeats 5

Prints markdown tables (copy them into notes or a PR) and saves eval/results/retrieval-<timestamp>.json with every query's rank.
Needs the database (and Redis for the embedding cache); it never calls Gemini."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app` and `eval` import when run as a script

from eval import metrics  # noqa: E402

MODE_NAMES = ("vector", "keyword", "hybrid", "hybrid+rerank")


class ModeUnavailable(Exception):
    """A mode cannot run here (for example the reranker model is not downloaded): reported, not fatal."""


def build_modes(skip_rerank: bool = False) -> dict:
    """mode name -> fn(db, query, k) -> hits. highlight=False everywhere: it is what RAG and the agent use, and it keeps
    sentence embedding out of the latency numbers."""
    from app.services import retrieval

    def rerank_checked(db, q, k):
        hits = retrieval.hybrid_search(db, q, k=k, highlight=False, rerank=True)
        if len(hits) >= 2 and hits[0].get("rerank_score") is None:  # the service silently falls back to the fused order
            raise ModeUnavailable("the reranker could not run (model not downloaded?): see the log for the reason")
        return hits

    modes = {
        "vector": lambda db, q, k: retrieval.vector_search(db, q, k=k, highlight=False),
        "keyword": lambda db, q, k: retrieval.keyword_search(db, q, k=k, highlight=False),
        "hybrid": lambda db, q, k: retrieval.hybrid_search(db, q, k=k, highlight=False, rerank=False),
        "hybrid+rerank": rerank_checked,
    }
    if skip_rerank:
        del modes["hybrid+rerank"]
    return modes


def run_retrieval_eval(db, queries: list, modes: dict, *, k: int = 10, repeats: int = 3, clock=time.perf_counter, progress=print) -> dict:
    """For every mode and query: one untimed warm-up call (fills the query-embedding cache and loads models, so latency is the
    steady-state cost), then `repeats` timed calls; the query's latency is their median. Returns {mode: {...}, 'skipped': {...}}."""
    out: dict = {"modes": {}, "skipped": {}}
    for name, search in modes.items():
        per_query = []
        try:
            for q in queries:
                search(db, q.query, k)  # warm-up
                times, hits = [], []
                for _ in range(repeats):
                    started = clock()
                    hits = search(db, q.query, k)
                    times.append((clock() - started) * 1000)
                    if hasattr(db, "rollback"):
                        db.rollback()
                per_query.append({
                    "query": q.query, "type": q.type, "rank": metrics.first_hit_rank(hits, q.gold), "latency_ms": metrics.median(times),
                    "top": [{"video_id": h["video_id"], "start_sec": h["start_sec"], "end_sec": h["end_sec"], "score": h.get("score")} for h in hits[:3]],
                })
        except ModeUnavailable as exc:
            out["skipped"][name] = str(exc)
            progress(f"  {name}: skipped ({exc})")
            continue
        by_type = {t: metrics.retrieval_summary([r["rank"] for r in per_query if r["type"] == t], [r["latency_ms"] for r in per_query if r["type"] == t])
                   for t in metrics.QUERY_TYPES if any(r["type"] == t for r in per_query)}
        out["modes"][name] = {"summary": metrics.retrieval_summary([r["rank"] for r in per_query], [r["latency_ms"] for r in per_query]),
                              "by_type": by_type, "per_query": per_query}
        progress(f"  {name}: done")
    return out


def _row(name: str, s: dict) -> list[str]:
    return [name, str(s["n"]), metrics.fmt(s["recall@1"], "pct"), metrics.fmt(s["recall@5"], "pct"), metrics.fmt(s["mrr"], "ratio"),
            metrics.fmt(s["median_latency_ms"], "ms")]


def render_report(results: dict, k: int) -> str:
    headers = ["Mode", "Queries", "Recall@1", "Recall@5", f"MRR@{k}", "Median latency"]
    parts = ["### All queries", metrics.markdown_table(headers, [_row(n, m["summary"]) for n, m in results["modes"].items()])]
    for t in metrics.QUERY_TYPES:
        rows = [_row(n, m["by_type"][t]) for n, m in results["modes"].items() if t in m["by_type"]]
        if rows:
            parts += ["", f"### `{t}` queries", metrics.markdown_table(headers, rows)]
    for name, why in results["skipped"].items():
        parts.append(f"\n_{name}: skipped, {why}_")
    return "\n".join(parts)


def warnings(results: dict) -> list[str]:
    """The Phase 12 expectation: hybrid should not lose to vector on exact-word queries. Say so loudly when it does."""
    out = []
    v, h = results["modes"].get("vector"), results["modes"].get("hybrid")
    if v and h and "exact" in v["by_type"] and "exact" in h["by_type"]:
        if h["by_type"]["exact"]["recall@5"] < v["by_type"]["exact"]["recall@5"]:
            out.append("hybrid has LOWER Recall@5 than vector on `exact` queries: investigate before trusting hybrid (look at per_query in the JSON).")
    return out


def save_results(results: dict, config: dict, directory: Path = metrics.RESULTS_DIR, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"retrieval-{now.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"timestamp": now.isoformat(), "config": config, **results}, indent=2), encoding="utf-8")
    return path


def probe_reranker() -> str | None:
    """None when the cross-encoder loads and scores; otherwise why not (so the run can skip it up front, once)."""
    try:
        from app.services import rerank
        rerank.score_pairs("probe", ["probe"])
        return None
    except Exception as exc:  # download blocked, offline, out of memory...
        return f"{type(exc).__name__}: {str(exc)[:120]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", type=Path, default=metrics.DEFAULT_QUERIES)
    ap.add_argument("--k", type=int, default=10, help="results fetched per query (Recall@5 needs at least 5)")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per query and mode (the median is used)")
    ap.add_argument("--skip-rerank", action="store_true", help="do not run hybrid+rerank (avoids downloading the cross-encoder)")
    ap.add_argument("--out", type=Path, default=metrics.RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.k < 5 or args.repeats < 1:
        ap.error("--k must be at least 5 and --repeats at least 1")

    if not args.queries.exists():
        print(f"{args.queries} does not exist. Create it with: docker compose exec api python eval/add_query.py")
        return 1
    try:
        queries = metrics.load_queries(args.queries)
    except ValueError as exc:
        print(f"Bad gold file: {exc}")
        return 1
    if not queries:
        print(f"{args.queries} has no queries yet.")
        return 1

    from sqlalchemy import select

    from app.core.config import settings
    from app.core.db import SessionLocal
    from app.models import Video

    with SessionLocal() as db:
        known = {str(v) for v in db.scalars(select(Video.id))}
        missing = sorted({g.video_id for q in queries for g in q.gold} - known)
        if missing:
            print(f"WARNING: {len(missing)} gold video id(s) are not in the database (queries on them can never hit): {', '.join(missing)}")
        counts = {t: sum(1 for q in queries if q.type == t) for t in metrics.QUERY_TYPES}
        videos = len({g.video_id for q in queries for g in q.gold})
        print(f"{len(queries)} queries ({', '.join(f'{n} {t}' for t, n in counts.items())}) across {videos} videos; k={args.k}, repeats={args.repeats}")
        if len(queries) < 30 or videos < 5:
            print("NOTE: fewer than 30 queries or 5 videos: treat the numbers as a rough guide, not a verdict.")

        modes = build_modes(skip_rerank=args.skip_rerank)
        if "hybrid+rerank" in modes:
            why = probe_reranker()
            if why:
                del modes["hybrid+rerank"]
                print(f"hybrid+rerank skipped: reranker unavailable ({why})")
        results = run_retrieval_eval(db, queries, modes, k=args.k, repeats=args.repeats)
        if args.skip_rerank:
            results["skipped"]["hybrid+rerank"] = "--skip-rerank"

    print("\n" + render_report(results, args.k) + "\n")
    for w in warnings(results):
        print("WARNING:", w)
    config = {"queries_file": str(args.queries), "k": args.k, "repeats": args.repeats, "embed_model": settings.EMBED_MODEL,
              "min_score": settings.MIN_SCORE, "hybrid_candidates": settings.HYBRID_CANDIDATES, "rrf_k": settings.RRF_K,
              "rerank_model": settings.RERANK_MODEL, "rerank_candidates": settings.RERANK_CANDIDATES,
              "query_counts": counts, "videos": videos,
              "latency_note": "median of repeated warm calls per query; query embedding cached, highlight off"}
    print("saved", save_results(results, config, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
