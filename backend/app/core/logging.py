"""Application logging: one stdout handler on the root logger, level from LOG_LEVEL.

Why: under gunicorn + UvicornWorker (and in Celery) nobody configures the root logger, so it stays at WARNING and every
`log.info(...)` in the app is dropped. Containers should log to stdout; Docker's json-file driver does the rotation.

Access logs are NOT duplicated: uvicorn's and gunicorn's access/error loggers have their own handlers and `propagate=False`,
so records sent to them never reach the root handler installed here."""
import logging
import sys

from app.core.config import settings

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
_MARK = "_svs_stdout_handler"
# Chatty libraries: their INFO lines (every S3 / HTTP request) would drown the app's own. They stay at WARNING unless DEBUG.
_QUIET = ("botocore", "boto3", "s3transfer", "urllib3", "httpx", "httpcore")
# Loggers that own their handlers; they must never also write through the root handler.
_OWN_HANDLERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error", "gunicorn.access")


class _StdoutHandler(logging.StreamHandler):
    """Looks `sys.stdout` up at emit time, so a stream swapped later (pytest capture, a supervisor) never leaves the
    handler writing into a closed file."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:
        pass


def setup_logging(level: str | None = None) -> None:
    """Idempotent: calling it again (several workers, Celery signals, reloads) never adds a second handler."""
    name = (level or settings.LOG_LEVEL).upper()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARK, False):
            root.removeHandler(handler)
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    setattr(handler, _MARK, True)
    root.addHandler(handler)
    root.setLevel(name)

    for noisy in _QUIET:
        logging.getLogger(noisy).setLevel(logging.DEBUG if name == "DEBUG" else logging.WARNING)
    for own in _OWN_HANDLERS:
        lg = logging.getLogger(own)
        if lg.handlers:  # configured by uvicorn/gunicorn: keep it out of the root handler
            lg.propagate = False
