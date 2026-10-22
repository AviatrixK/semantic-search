# Evaluation harness

Numbers instead of opinions: a hand-labelled set of questions, and scripts that measure how well search does on it.
Run everything inside the api container (it has the database, Redis and the models):

```powershell
docker compose exec api python eval/add_query.py              # 1. label queries while you watch a video
docker compose exec api python eval/run_eval.py               # 2. retrieval: vector vs keyword vs hybrid vs hybrid+rerank
```
`backend/` is mounted into the container, so `eval/queries.jsonl` and `eval/results/*.json` appear on your disk.

## The gold file: `eval/queries.jsonl`
One JSON object per line:
`{"query": "...", "video_id": "<uuid>", "start_sec": 123.0, "end_sec": 160.0, "type": "exact|paraphrase|multi"}`

| type | what to write | tests |
|---|---|---|
| `exact` | a question using words actually spoken (names, acronyms, a distinctive phrase) | keyword/hybrid strength |
| `paraphrase` | the same idea in words the speaker did **not** use | semantic/vector strength |
| `multi` | a question needing more than one moment ("compare what is said about X early and late") | several gold moments per question |

`add_query.py` shows the transcript of the moment you type (`2:03-2:40`, `123-160`, `1m30s-2m`) so you can check the label, warns if the
range is past the end of the video, and lets you undo (`u`) the last entry. **Writing the same question again adds another correct
moment to it**: that is how a `multi` question gets two gold ranges (also across videos). Aim for 30+ queries over 5+ videos; fewer is
only a rough guide, and the scripts say so.

## Retrieval metrics (`run_eval.py`)
A result is a **hit** when it is in the same video and its time range overlaps a gold range (touching ends do not count). The rank
of the first hit is what all metrics use. Each query is searched with the top `--k` (default 10).

- **Recall@1**: share of queries whose very first result is a hit. "Did it get it right immediately?"
- **Recall@5**: share with a hit anywhere in the top 5. "Would a person scanning five results find it?"
- **MRR@k** (mean reciprocal rank): the average of 1/rank of the first hit (a miss counts 0). 1.0 = always first, 0.5 = typically second.
  Rewards putting the right moment higher, which Recall@5 ignores.
- **Median latency**: the median over queries of each query's median of `--repeats` warm calls. The embedding is cached and highlights are
  off, so this is the steady-state search cost; a never-seen query adds ~20-100 ms for the embedding (vector, hybrid, rerank only).

Modes: `vector` (meaning), `keyword` (Postgres full-text), `hybrid` (both, fused with RRF), `hybrid+rerank` (hybrid, then a cross-encoder
rescores the top 20). Tables are printed overall and per query type. `hybrid+rerank` downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`
(~90 MB) on first use; use `--skip-rerank` to avoid that, or it is skipped (and reported) if the model cannot load.
Expectation: hybrid >= vector on `exact`; the script warns when it is not.

## Comparing embedding providers (`run_provider_eval.py`)
Evaluates each provider on ITS OWN stored vectors (they live side by side in `chunk_embeddings`, one set per `provider/model@dims`),
on the same gold queries and the same chunks, so the only thing that differs is the model.

```powershell
# 1. fill the vectors of every provider you want to compare (embed-only: no re-transcription; resumable; --dry-run shows the cost first)
docker compose exec api python -m app.scripts.reprocess --embed-provider gemini --dry-run
docker compose exec api python -m app.scripts.reprocess --embed-provider gemini
# 2. compare (asks before making real API calls: one query embedding per gold query and provider)
docker compose exec api python eval/run_provider_eval.py --providers local,gemini,cohere,openai
docker compose exec api python eval/run_provider_eval.py --inr-per-usd 88 --assume-both-variants
```
It prints Recall@1, Recall@5, MRR and median search latency per provider and mode (vector, hybrid), the cold latency (first,
uncached search, includes the provider round trip), and the estimated cost per 1,000 searches and per hour of video.
Saves `eval/results/providers-<timestamp>.json`.

- A provider that is not configured, or has no vectors yet, is skipped with the reason. One that fails or times out DURING the run is
  reported as invalid, not scored: search would silently fall back to keyword-only for it, which would make the numbers meaningless.
- Cost uses list prices (`app/services/pricing.py`, with source links and dates) and about 4 characters per token unless the provider
  reports tokens. Indic scripts use more tokens per character, so real costs for Hindi and similar are higher.
- Sarvam transcription is shown separately in INR per hour of video: ONE speech-to-text job (INR 30/h) plus INR 20 per 10K characters
  for each text version (romanised, English) made from the chunk text; the characters per hour come from the corpus being evaluated. No
  exchange rate is assumed: pass `--inr-per-usd` to get a combined USD total.
- `MIN_SCORE` (0.25) was tuned for the local model; other models spread their cosine similarities differently, so each provider has its
  own `MIN_SCORE_<PROVIDER>` (empty = `MIN_SCORE`); the JSON records the value used for every provider. If a provider shows a
  suspiciously low Recall, look at `per_query` in the JSON before concluding it is a worse model.

## Results
Each run saves `eval/results/retrieval-<timestamp>.json` with the settings used (models, `MIN_SCORE`,
RRF k...) and every per-query outcome, so a miss can be inspected and runs can be compared over time.
