"""setup_logging(): INFO reaches stdout (the root logger is WARNING by default under gunicorn), in the agreed format, once,
without duplicating access logs; wired into API startup and the Celery worker/beat."""
import asyncio
import logging
import re
import sys
from unittest.mock import patch

import pytest

from app import main
from app.core import logging as applog
from app.core.config import Settings
from app.services import embedding, storage

LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d INFO app\.demo hello world$")


@pytest.fixture(autouse=True)
def clean_root():
    """setup_logging mutates the root logger: put it back exactly as it was."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    quiet = {n: logging.getLogger(n).level for n in applog._QUIET}
    own = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate) for n in applog._OWN_HANDLERS}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for n, lvl in quiet.items():
        logging.getLogger(n).setLevel(lvl)
    for n, (hs, prop) in own.items():
        logging.getLogger(n).handlers[:] = hs
        logging.getLogger(n).propagate = prop


def root_default():
    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # what gunicorn/celery leave behind
    root.handlers[:] = []


def test_info_is_lost_by_default_and_visible_after_setup(capsys):
    root_default()
    logging.getLogger("app.demo").info("hello world")
    assert capsys.readouterr().out == ""  # the bug this fixes
    applog.setup_logging("INFO")
    logging.getLogger("app.demo").info("hello world")
    assert LINE.match(capsys.readouterr().out.strip())  # time level logger message, on stdout


def test_goes_to_stdout_not_stderr(capsys):
    root_default()
    applog.setup_logging("INFO")
    logging.getLogger("app.demo").warning("careful")
    out = capsys.readouterr()
    assert "WARNING app.demo careful" in out.out and out.err == ""


def test_level_comes_from_the_log_level_setting(capsys):
    root_default()
    with patch.object(applog.settings, "LOG_LEVEL", "WARNING"):
        applog.setup_logging()
    logging.getLogger("app.demo").info("hidden")
    logging.getLogger("app.demo").warning("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out and "shown" in out
    assert logging.getLogger().level == logging.WARNING


def test_calling_it_twice_installs_one_handler(capsys):
    root_default()
    applog.setup_logging("INFO")
    applog.setup_logging("INFO")
    marked = [h for h in logging.getLogger().handlers if getattr(h, applog._MARK, False)]
    assert len(marked) == 1
    logging.getLogger("app.demo").info("hello world")
    assert capsys.readouterr().out.count("hello world") == 1  # not twice


def test_access_logs_are_not_duplicated(capsys):
    """gunicorn / uvicorn access loggers write through their own handler; the root handler must not repeat the line."""
    root_default()
    sink = logging.StreamHandler(sys.stderr)
    for name in ("uvicorn.access", "gunicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = [sink]
        lg.propagate = True  # worst case: something re-enabled propagation
    applog.setup_logging("INFO")
    for name in ("uvicorn.access", "gunicorn.access"):
        assert logging.getLogger(name).propagate is False
        logging.getLogger(name).info('127.0.0.1 - "GET /health HTTP/1.1" 200')
    out = capsys.readouterr()
    assert "GET /health" not in out.out  # only the access logger's own handler (stderr) wrote it
    assert out.err.count("GET /health") == 2


def test_chatty_libraries_stay_quiet_unless_debug():
    root_default()
    applog.setup_logging("INFO")
    assert logging.getLogger("botocore").level == logging.WARNING
    applog.setup_logging("DEBUG")
    assert logging.getLogger("botocore").level == logging.DEBUG


def test_survives_the_stream_being_replaced(capsys):
    """pytest (and supervisors) swap sys.stdout: the handler must follow it instead of writing to a closed file."""
    root_default()
    applog.setup_logging("INFO")
    import io
    replacement = io.StringIO()
    with patch.object(sys, "stdout", replacement):
        logging.getLogger("app.demo").info("hello world")
    assert "hello world" in replacement.getvalue()


@pytest.mark.parametrize("value,expected", [("debug", "DEBUG"), (" warning ", "WARNING"), ("INFO", "INFO")])
def test_log_level_setting_is_normalised(value, expected):
    assert Settings(LOG_LEVEL=value).LOG_LEVEL == expected


def test_default_is_info_and_nonsense_is_rejected():
    assert Settings.model_fields["LOG_LEVEL"].default == "INFO"
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        Settings(LOG_LEVEL="loud")


def test_api_configures_logging_first_thing_at_startup():
    order = []

    async def go():
        async with main.lifespan(main.app):
            pass

    with patch.object(main, "setup_logging", side_effect=lambda: order.append("logging")), \
         patch.object(storage, "ensure_bucket", side_effect=lambda: order.append("bucket")), \
         patch.object(embedding, "warm_up", side_effect=lambda: order.append("warm") or 1.0):
        asyncio.run(go())
    assert order == ["logging", "bucket", "warm"]


def test_celery_worker_and_beat_use_it_through_the_setup_logging_signal():
    """One signal covers `celery worker` and `celery beat`: connecting a receiver makes Celery skip its own setup."""
    from celery.signals import setup_logging as celery_setup_logging

    from app.workers import celery_app
    assert any(r[1]() is celery_app._configure_logging for r in celery_setup_logging.receivers if r[1]() is not None)
    with patch.object(applog, "setup_logging") as setup:
        celery_setup_logging.send(sender=None, loglevel=20, logfile=None, format="", colorize=False)
    setup.assert_called_once_with()
