"""One real, tiny request to Gemini using the settings from .env: confirms the key, the model name and the network path.
This DOES call the Gemini API (a few tokens), so run it on purpose.

Usage:
    docker compose exec api python -m app.scripts.check_llm            # sends: Reply with the single word: OK
    docker compose exec api python -m app.scripts.check_llm --models   # lists model ids this key can use for generateContent
"""
import sys

from app.core.config import settings
from app.services.llm import LLM, LLMError, get_llm


def list_models() -> list[str]:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return sorted(m.name.removeprefix("models/") for m in client.models.list() if "generateContent" in (m.supported_actions or []))


def main(argv: list[str], llm: LLM | None = None) -> int:
    if "--models" in argv:
        if not settings.GEMINI_API_KEY:
            print("GEMINI_API_KEY is not set in .env")
            return 1
        try:
            print("\n".join(list_models()))
        except Exception as exc:  # the SDK message can echo request details: show the type only
            print(f"Could not list models: {type(exc).__name__}")
            return 1
        return 0
    try:
        r = (llm or get_llm()).generate(system="Reply with exactly one word.", prompt="Reply with the single word: OK")
    except LLMError as exc:
        print(f"FAILED: {exc.user_message}  [{type(exc).__name__}: {exc}]")
        return 1
    print(f"OK  model={r.model}  latency={r.latency_ms} ms  tokens: prompt={r.prompt_tokens} completion={r.completion_tokens} total={r.total_tokens}")
    print(f"reply: {r.text[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
