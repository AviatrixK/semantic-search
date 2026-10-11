"""docker-compose.prod.yml: every service rotates its container logs (json-file, 10 MB x 3), so logs can never fill the disk.
api, worker and beat get it through the shared `*backend` anchor; this test checks the RESOLVED config, so a service added
later without logging fails here."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.prod.yml"


@pytest.fixture(scope="module")
def services():
    if not COMPOSE.exists():  # inside the api container only ./backend is mounted
        pytest.skip(f"{COMPOSE} not available here")
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


def test_every_service_rotates_its_logs(services):
    assert set(services) >= {"postgres", "redis", "api", "worker", "beat", "caddy"}
    for name, svc in services.items():
        logging = svc.get("logging")
        assert logging, f"{name} has no logging section"
        assert logging["driver"] == "json-file", name
        assert logging["options"] == {"max-size": "10m", "max-file": "3"}, name
