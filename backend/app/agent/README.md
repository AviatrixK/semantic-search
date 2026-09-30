# Agent (weeks 7-9)

Planned files:
- `tools.py`  — tool schemas + executors: `search_transcripts`, `list_videos`, `get_transcript_window`.
  Each executor calls `app.services.retrieval` — never raw SQL here.
- `loop.py`   — plan → act → observe loop, `max_steps=5`, returns `{answer, citations, trace}`.
- `prompts.py` — system prompt: answer ONLY from tool results, cite as `[video_id@start_sec]`.
- `router.py` — decide simple search vs. agent.

Route: `POST /api/ask` in `app/api/routes/ask.py`, protected by `current_user` and rate-limited.
