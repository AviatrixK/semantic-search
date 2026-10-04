from app.core.config import settings
from app.scripts import check_llm
from app.services.llm import LLMNotConfigured, LLMRejected
from tests.fakes import FakeLLM


def test_success_prints_model_latency_and_tokens(capsys):
    assert check_llm.main([], llm=FakeLLM("OK")) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK  model=fake") and "total=2" in out and "reply: 'OK'" in out


def test_failure_prints_the_reason_and_exits_non_zero(capsys):
    assert check_llm.main([], llm=FakeLLM(LLMNotConfigured("no key"))) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "GEMINI_API_KEY" in out
    assert check_llm.main([], llm=FakeLLM(LLMRejected("404 model"))) == 1


def test_models_flag_without_a_key_says_so_and_makes_no_call(capsys, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(check_llm, "list_models", lambda: (_ for _ in ()).throw(AssertionError("must not call the API")))
    assert check_llm.main(["--models"]) == 1
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().out


def test_models_flag_lists_names_and_hides_error_details(capsys, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(check_llm, "list_models", lambda: ["gemini-a", "gemini-b"])
    assert check_llm.main(["--models"]) == 0
    assert capsys.readouterr().out.split() == ["gemini-a", "gemini-b"]

    def boom():
        raise RuntimeError("key=SECRET")

    monkeypatch.setattr(check_llm, "list_models", boom)
    assert check_llm.main(["--models"]) == 1
    assert "SECRET" not in capsys.readouterr().out
