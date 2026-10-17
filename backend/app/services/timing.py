"""Per-request timing for search, reported as response headers (see api/routes/search.py).

Passing a `Timings` through the retrieval functions costs nothing when it is None, so other callers and the
eval harness are unchanged."""
import time
from contextlib import contextmanager, nullcontext


class Timings:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - started) * 1000

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def server_timing(self, total_ms: float) -> str:
        """The standard `Server-Timing` header: browsers show each metric in DevTools > Network > Timing."""
        parts = [f"{name};dur={value:.1f}" for name, value in self.ms.items()]
        parts.append(f"total;dur={total_ms:.1f}")
        return ", ".join(parts)

    def highlight_summary(self) -> str:
        """e.g. "stored=8, fallback=0, single=2": where each result's highlight came from."""
        return ", ".join(f"{k}={self.counts.get(k, 0)}" for k in ("stored", "fallback", "single"))


def measure(timings: "Timings | None", name: str):
    """`timings.measure(name)`, or a no-op when nobody is measuring."""
    return timings.measure(name) if timings is not None else nullcontext()
