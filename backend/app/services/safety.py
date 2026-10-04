"""Prompt-injection hygiene. Transcript text is written by whoever made the video, so it is DATA, never instructions.
Two layers: the prompts say so explicitly, and here we make sure the data cannot break out of the tags that mark it."""
import re

_SPACES = re.compile(r"\s+")


def sanitize_untrusted(text: str) -> str:
    """One line, with angle brackets turned into look-alike characters so text can never contain a closing tag such as
    </transcript_excerpt>, </excerpts> or </question> and escape the block it is wrapped in."""
    return _SPACES.sub(" ", text).strip().replace("<", "‹").replace(">", "›")


def wrap_excerpt(text: str) -> str:
    """Transcript text as the agent's tools return it."""
    return f"<transcript_excerpt>{sanitize_untrusted(text)}</transcript_excerpt>"
