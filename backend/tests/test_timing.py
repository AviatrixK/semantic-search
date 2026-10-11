from app.services.timing import Timings, measure


def test_measure_accumulates_and_formats_server_timing():
    t = Timings()
    with t.measure("sql"):
        pass
    with t.measure("sql"):
        pass
    with t.measure("embed"):
        pass
    assert list(t.ms) == ["sql", "embed"]  # insertion order, repeated names are summed
    header = t.server_timing(12.34)
    assert header.startswith("sql;dur=") and ", embed;dur=" in header and header.endswith("total;dur=12.3")


def test_highlight_summary_defaults_to_zero():
    t = Timings()
    t.count("stored", 8)
    t.count("single", 2)
    assert t.highlight_summary() == "stored=8, fallback=0, single=2"


def test_measure_is_a_noop_without_timings():
    with measure(None, "anything"):
        pass
    t = Timings()
    with measure(t, "x"):
        pass
    assert "x" in t.ms


def test_timing_is_recorded_even_when_the_step_raises():
    t = Timings()
    try:
        with t.measure("boom"):
            raise ValueError
    except ValueError:
        pass
    assert "boom" in t.ms
