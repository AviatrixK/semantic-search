"""Sarvam against a FAKE Sarvam API (mocked HTTP transport, same REST flow as the real one): ONE batch speech-to-text job per video, the
parameters sent, the upload headers, polling, parsing timestamps, splitting long audio, resuming a saved job instead of starting a second
one, saaras:v4 with the v3 fallback, and the text APIs (/transliterate, /translate) that make the romanised and English versions.
No network, no key."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from app.core.config import settings
from app.services import media, sarvam
from app.services.chunking import ChunkSpan
from app.services.sarvam import (SarvamClient, SarvamJobFailed, SarvamNoTimestamps, SarvamNotConfigured, SarvamRejected, SarvamUnavailable,
                                 add_text_variants, parse_output, split_for_limit, text_plan, transcribe_audio)
from app.services.transcription import NoSpeechError

NATIVE = {"transcript": "नमस्ते दोस्तों आज हम बात करेंगे", "language_code": "hi-IN", "timestamps": {
    "chunks": ["नमस्ते दोस्तों", "आज हम बात करेंगे"], "start_time_seconds": [0.0, 8.0], "end_time_seconds": [7.5, 19.0]}}
API = "api.sarvam.ai"


class FakeSarvam:
    """The REST flow of Sarvam's batch API plus the two text APIs. `states` is the sequence of job_state values status polls return.
    `jobs` holds every job ever created (id -> parameters); `fail` maps (method, path) to an HTTP status (or a callable returning one)."""

    parts = 1

    def __init__(self, states=("Accepted", "Running", "Completed"), output=None, fail=None):
        self.states, self.output, self.fail = list(states), output or NATIVE, fail or {}
        self.calls, self.jobs, self.uploads, self.starts, self.polls, self.text_calls = [], {}, [], [], 0, []
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def created(self) -> int:
        return len([c for c in self.calls if c[0] == "POST" and c[2] == "/speech-to-text/job/v1"])

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, body = request.url.path, (json.loads(request.content) if request.content and request.method == "POST" else None)
        self.calls.append((request.method, request.url.host, path, request.headers.get("api-subscription-key"), body))
        failure = self.fail.get((request.method, path))
        if failure is not None:
            code = failure(body) if callable(failure) else failure
            if code:
                return httpx.Response(code)
        if request.method == "POST" and path == "/speech-to-text/job/v1":
            job_id = f"job-{len(self.jobs) + 1}"
            self.jobs[job_id] = body["job_parameters"]
            return httpx.Response(200, json={"job_id": job_id, "job_state": "Accepted", "storage_container_type": "Azure"})
        if path.endswith("/upload-files"):
            return httpx.Response(200, json={"job_id": body["job_id"], "upload_urls": {n: {"file_url": f"https://blob.test/in/{body['job_id']}/{n}"} for n in body["files"]}})
        if request.method == "PUT":
            self.uploads.append((request.url.path, request.headers.get("x-ms-blob-type"), request.headers.get("content-type"), len(request.content)))
            return httpx.Response(201)
        if path.endswith("/start"):
            self.starts.append(path.split("/")[-2])
            return httpx.Response(200, json={})
        if path.endswith("/status"):
            job_id = path.split("/")[-2]
            if job_id not in self.jobs:
                return httpx.Response(404)
            state = self.states[min(self.polls, len(self.states) - 1)]
            self.polls += 1
            details = [{"inputs": [{"file_name": f"part{i:03d}.mp3"}], "outputs": [{"file_name": f"{i}.json"}], "state": "Success"}
                       for i in range(self.parts)] if state == "Completed" else []
            return httpx.Response(200, json={"job_id": job_id, "job_state": state, "job_details": details, "error_message": "boom" if state == "Failed" else None})
        if path.endswith("/download-files"):
            return httpx.Response(200, json={"download_urls": {f: {"file_url": f"https://blob.test/out/{body['job_id']}/{f}"} for f in body["files"]}})
        if request.method == "GET" and request.url.host == "blob.test":
            return httpx.Response(200, json=self.output)
        if request.method == "POST" and path in ("/transliterate", "/translate"):
            self.text_calls.append((path, body))
            key = "transliterated_text" if path == "/transliterate" else "translated_text"
            return httpx.Response(200, json={key: ("R:" if path == "/transliterate" else "EN:") + body["input"], "source_language_code": body["source_language_code"]})
        return httpx.Response(404)


def client_for(fake, **kw):
    return SarvamClient(api_key=kw.pop("api_key", "sarvam-secret"), client=fake.client, sleep=lambda s: None, **kw)


@pytest.fixture
def audio(tmp_path):
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 100)
    return str(f)


@pytest.fixture
def one_part(audio):
    with patch.object(media, "duration_float", return_value=20.0), patch.object(media, "split_audio", return_value=[(audio, 0.0)]):
        yield


@pytest.fixture(autouse=True)
def models():
    with patch.object(settings, "SARVAM_MODEL", "saaras:v3"), patch.object(settings, "SARVAM_FALLBACK_MODEL", "saaras:v3"), \
         patch.object(settings, "SARVAM_V4_UNSUPPORTED", ""), patch.object(settings, "SARVAM_TEXT_VARIANTS", "roman,en"), \
         patch.object(settings, "SARVAM_TRANSLATE_MODEL", "sarvam-translate:v1"):
        yield


def run(fake, tmp_path, audio, **kw):
    saved = []
    result = transcribe_audio(audio, str(tmp_path), client=kw.pop("client", None) or client_for(fake), on_job=lambda j, s: saved.append((j, s)), **kw)
    return result, saved


# ---------------------------------------------------------------- parsing (pure)
def test_parse_output_reads_chunk_timestamps_and_adds_the_part_offset():
    segs = parse_output(NATIVE, offset=100.0)
    assert segs == [{"start": 100.0, "end": 107.5, "text": "नमस्ते दोस्तों"}, {"start": 108.0, "end": 119.0, "text": "आज हम बात करेंगे"}]


def test_silence_is_an_empty_list_but_text_without_timestamps_is_an_error():
    assert parse_output({"transcript": "", "timestamps": {"chunks": [], "start_time_seconds": [], "end_time_seconds": []}}) == []
    assert parse_output({}) == []
    with pytest.raises(SarvamNoTimestamps):
        parse_output({"transcript": "there is text", "timestamps": None})
    with pytest.raises(SarvamNoTimestamps):  # arrays of different lengths are not usable either
        parse_output({"transcript": "x", "timestamps": {"chunks": ["a", "b"], "start_time_seconds": [0.0], "end_time_seconds": [1.0, 2.0]}})


def test_the_three_job_alignment_is_gone():
    assert not hasattr(sarvam, "align_text") and not hasattr(sarvam, "merge_variants") and not hasattr(sarvam, "MODES")
    assert not hasattr(settings, "sarvam_outputs")


# ---------------------------------------------------------------- the REST flow: ONE job
def test_the_batch_flow_sends_the_documented_requests(one_part, audio, tmp_path):
    fake = FakeSarvam()
    run(fake, tmp_path, audio, language="hi-IN")
    ops = [(m, p) for m, h, p, k, b in fake.calls if h == API]
    assert ops == [("POST", "/speech-to-text/job/v1"), ("POST", "/speech-to-text/job/v1/upload-files"),
                   ("POST", "/speech-to-text/job/v1/job-1/start"), ("GET", "/speech-to-text/job/v1/job-1/status"),
                   ("GET", "/speech-to-text/job/v1/job-1/status"), ("GET", "/speech-to-text/job/v1/job-1/status"),
                   ("POST", "/speech-to-text/job/v1/download-files")]
    assert all(k == "sarvam-secret" for m, h, p, k, b in fake.calls if h == API)  # auth header on every API call
    assert all(k is None for m, h, p, k, b in fake.calls if h == "blob.test")  # and NEVER sent to the pre-signed storage URLs
    assert fake.jobs["job-1"] == {"model": "saaras:v3", "mode": "transcribe", "language_code": "hi-IN"}
    assert fake.uploads == [("/in/job-1/part000.mp3", "BlockBlob", "audio/mpeg", 103)]


def test_a_video_is_transcribed_by_exactly_one_job_in_the_native_script(one_part, audio, tmp_path):
    fake = FakeSarvam()
    result, saved = run(fake, tmp_path, audio)
    assert fake.created() == 1 and [j["mode"] for j in fake.jobs.values()] == ["transcribe"]  # no translit / translate audio jobs
    assert fake.jobs["job-1"]["language_code"] == "unknown"  # no language chosen: auto-detect
    assert result.billed_seconds == 20.0 == result.audio_seconds  # one job x 20 s: that is what Sarvam bills for the audio
    assert result.engine == "sarvam" and result.language == "hi-IN"
    assert [seg["text"] for seg in result.segments] == ["नमस्ते दोस्तों", "आज हम बात करेंगे"]
    assert all(set(seg) == {"start", "end", "text"} for seg in result.segments)  # the other scripts come from the text APIs, later


def test_the_job_id_is_reported_at_every_step_so_it_can_be_saved(one_part, audio, tmp_path):
    fake = FakeSarvam()
    _, saved = run(fake, tmp_path, audio)
    assert saved[:3] == [("job-1", "created"), ("job-1", "uploaded"), ("job-1", "started")]
    assert [s for _, s in saved[3:]] == ["accepted", "running", "completed"]  # then each new state Sarvam reports (once)
    assert {j for j, _ in saved} == {"job-1"}


def test_long_audio_is_split_and_every_part_lands_on_one_timeline_in_the_one_job(audio, tmp_path):
    fake = FakeSarvam()
    fake.parts = 2
    p1, p2 = str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")
    for p in (p1, p2):
        open(p, "wb").write(b"x" * 10)
    with patch.object(media, "duration_float", return_value=14000.0), patch.object(media, "split_audio", return_value=[(p1, 0.0), (p2, 6600.0)]):
        result, _ = run(fake, tmp_path, audio)
    assert fake.created() == 1
    assert [u[0] for u in fake.uploads] == ["/in/job-1/part000.mp3", "/in/job-1/part001.mp3"]
    assert [s["start"] for s in result.segments] == [0.0, 8.0, 6600.0, 6608.0]  # the second part is shifted by its offset
    assert result.billed_seconds == 14000.0


def test_progress_is_reported_from_start_to_finish(one_part, audio, tmp_path):
    seen = []
    transcribe_audio(audio, str(tmp_path), on_progress=lambda done, total: seen.append(round(done / total, 2)), client=client_for(FakeSarvam()))
    assert seen[0] == 0.1 and seen[-1] == 1.0 and seen == sorted(seen)


def test_audio_with_no_speech_raises_no_speech(one_part, audio, tmp_path):
    silent = {"transcript": "", "language_code": "hi-IN", "timestamps": {"chunks": [], "start_time_seconds": [], "end_time_seconds": []}}
    with pytest.raises(NoSpeechError):
        run(FakeSarvam(output=silent), tmp_path, audio)


def test_a_reply_without_timestamps_stops_ingestion_with_a_clear_message(one_part, audio, tmp_path):
    with pytest.raises(SarvamNoTimestamps) as e:
        run(FakeSarvam(output={"transcript": "text but no timing", "language_code": "hi-IN"}), tmp_path, audio)
    assert "no timestamps" in e.value.user_message


# ---------------------------------------------------------------- failures
def test_a_failed_job_raises_with_a_safe_message_and_is_recorded_as_failed(one_part, audio, tmp_path):
    saved = []
    with pytest.raises(SarvamJobFailed) as e:
        transcribe_audio(audio, str(tmp_path), client=client_for(FakeSarvam(states=("Running", "Failed"))), on_job=lambda j, s: saved.append(s))
    assert "boom" not in e.value.user_message  # the provider's text stays in the log
    assert saved[-1] == "failed"


def test_a_job_that_never_finishes_is_unavailable_so_the_worker_retries(audio):
    ticks = iter(range(0, 10_000_000, 4000))
    fake = FakeSarvam(states=("Running",))
    fake.jobs["job-x"] = {}
    c = SarvamClient(api_key="k", client=fake.client, sleep=lambda s: None, clock=lambda: next(ticks))
    with pytest.raises(SarvamUnavailable):
        c.wait("job-x")


def test_http_errors_are_classified_and_never_leak_the_key(one_part, audio, tmp_path):
    fake = FakeSarvam(fail={("POST", "/speech-to-text/job/v1"): 401})
    with pytest.raises(SarvamRejected) as e:
        run(fake, tmp_path, audio)
    assert "sarvam-secret" not in str(e.value) and "sarvam-secret" not in e.value.user_message
    assert e.value.status == 401
    assert fake.created() == 1  # a 401 is not retried
    fake = FakeSarvam(fail={("POST", "/speech-to-text/job/v1"): 503})
    with pytest.raises(SarvamUnavailable):
        run(fake, tmp_path, audio)
    assert fake.created() == 3  # 1 try + 2 retries


def test_a_missing_key_fails_before_any_request(audio, tmp_path):
    fake = FakeSarvam()
    with pytest.raises(SarvamNotConfigured, match="SARVAM_API_KEY"):
        client_for(fake, api_key="").create_job("unknown")
    assert fake.calls == []


def test_upload_failures_are_classified(one_part, audio, tmp_path):
    for code, expected in ((403, SarvamRejected), (500, SarvamUnavailable)):
        fake = FakeSarvam()
        original = fake.handle
        fake.client = httpx.Client(transport=httpx.MockTransport(lambda r, o=original, c=code: httpx.Response(c) if r.method == "PUT" else o(r)))
        with pytest.raises(expected):
            run(fake, tmp_path, audio)


def test_every_sarvam_error_has_a_short_user_message():
    for cls in (sarvam.SarvamError, SarvamNotConfigured, SarvamUnavailable, SarvamRejected, SarvamJobFailed, SarvamNoTimestamps):
        assert 20 < len(cls("internal detail with a path C:/secret").user_message) < 160
        assert "secret" not in cls("internal detail with a path C:/secret").user_message


def test_a_short_outage_while_polling_does_not_abort_the_job(one_part, audio, tmp_path):
    fake = FakeSarvam()
    real, failures = fake.handle, {"left": 4}

    def flaky(request):
        if request.url.path.endswith("/status") and failures["left"] > 0:
            failures["left"] -= 1
            return httpx.Response(503)
        return real(request)

    fake.client = httpx.Client(transport=httpx.MockTransport(flaky))
    result, _ = run(fake, tmp_path, audio)
    assert result.segments and failures["left"] == 0
    assert len(fake.jobs) == 1  # one job was ever created: nothing was restarted


# ---------------------------------------------------------------- idempotency: a retry resumes the saved job
def test_a_failure_after_job_creation_does_not_create_a_second_job(one_part, audio, tmp_path):
    """The upload fails right after the job exists. The first attempt reports the job id; the retry is handed it back and must
    continue with THAT job: same upload, same start, and no new job (which Sarvam would bill again)."""
    fake = FakeSarvam(fail={("POST", "/speech-to-text/job/v1/upload-files"): lambda body: 503 if not fake.uploads and fake.first_try else 0})
    fake.first_try = True
    saved = []
    with pytest.raises(SarvamUnavailable):
        transcribe_audio(audio, str(tmp_path), client=client_for(fake), on_job=lambda j, s: saved.append((j, s)))
    assert saved == [("job-1", "created")] and fake.created() == 1

    fake.first_try = False  # the service is back; the worker retries the whole video
    result, saved2 = run(fake, tmp_path, audio, job_id="job-1", job_status="created")
    assert fake.created() == 1 and len(fake.jobs) == 1, "no second job may be created"
    assert fake.uploads == [("/in/job-1/part000.mp3", "BlockBlob", "audio/mpeg", 103)] and fake.starts == ["job-1"]
    assert result.segments and {j for j, _ in saved2} == {"job-1"}


def test_a_failure_after_the_job_started_resumes_polling_only(one_part, audio, tmp_path):
    fake = FakeSarvam(states=("Running", "Running", "Completed"))
    fake.jobs["job-1"] = {"model": "saaras:v3", "mode": "transcribe", "language_code": "hi-IN"}  # created, uploaded, started earlier
    result, _ = run(fake, tmp_path, audio, job_id="job-1", job_status="started")
    assert fake.created() == 0 and fake.uploads == [] and fake.starts == []  # nothing is created, uploaded or started again
    assert result.segments


def test_a_job_that_finished_while_the_worker_was_down_is_just_downloaded(one_part, audio, tmp_path):
    fake = FakeSarvam(states=("Completed",))
    fake.jobs["job-1"] = {}
    result, saved = run(fake, tmp_path, audio, job_id="job-1", job_status="running")
    assert fake.created() == 0 and fake.starts == [] and result.segments
    assert saved[-1] == ("job-1", "completed")


def test_a_crash_between_upload_and_start_only_starts(one_part, audio, tmp_path):
    fake = FakeSarvam()
    fake.jobs["job-1"] = {}
    run(fake, tmp_path, audio, job_id="job-1", job_status="uploaded")
    assert fake.created() == 0 and fake.uploads == [] and fake.starts == ["job-1"]


def test_a_start_that_did_reach_sarvam_is_not_repeated(one_part, audio, tmp_path):
    """Saved status is still "uploaded" (the reply to start was lost) but Sarvam already runs the job: do not start it again."""
    fake = FakeSarvam(states=("Running", "Completed"))
    fake.jobs["job-1"] = {}
    run(fake, tmp_path, audio, job_id="job-1", job_status="uploaded")
    assert fake.starts == [] and fake.created() == 0


@pytest.mark.parametrize("status", ["consumed", "failed", None, ""])
def test_a_consumed_or_failed_or_missing_job_is_never_resumed(one_part, audio, tmp_path, status):
    fake = FakeSarvam()
    fake.jobs["job-old"] = {}
    run(fake, tmp_path, audio, job_id="job-old", job_status=status)
    assert fake.created() == 1 and "job-old" not in fake.starts  # a new job, not the saved one


def test_a_saved_job_that_failed_on_sarvams_side_is_replaced_by_a_new_one(one_part, audio, tmp_path):
    fake = FakeSarvam(states=("Failed", "Accepted", "Completed"))
    fake.jobs["job-old"] = {}
    result, saved = run(fake, tmp_path, audio, job_id="job-old", job_status="running")
    assert fake.created() == 1 and fake.starts == ["job-2"] and result.segments


def test_a_saved_job_sarvam_no_longer_knows_is_replaced_by_a_new_one(one_part, audio, tmp_path):
    fake = FakeSarvam()  # job-gone is not in fake.jobs: the status call answers 404
    result, _ = run(fake, tmp_path, audio, job_id="job-gone", job_status="started")
    assert fake.created() == 1 and result.segments


def test_an_outage_while_checking_the_saved_job_keeps_it_for_the_next_retry(one_part, audio, tmp_path):
    fake = FakeSarvam(fail={("GET", "/speech-to-text/job/v1/job-1/status"): 503})
    fake.jobs["job-1"] = {}
    with pytest.raises(SarvamUnavailable):
        run(fake, tmp_path, audio, job_id="job-1", job_status="started")
    assert fake.created() == 0  # unreachable is not "gone": no second job


# ---------------------------------------------------------------- saaras:v4 with the v3 fallback
def test_v4_is_used_when_configured_and_the_language_is_supported(one_part, audio, tmp_path):
    fake = FakeSarvam()
    with patch.object(settings, "SARVAM_MODEL", "saaras:v4"):
        run(fake, tmp_path, audio, language="hi-IN")
    assert fake.jobs["job-1"]["model"] == "saaras:v4"


def test_languages_v4_does_not_support_go_to_v3_automatically(one_part, audio, tmp_path):
    fake = FakeSarvam()
    with patch.object(settings, "SARVAM_MODEL", "saaras:v4"), patch.object(settings, "SARVAM_V4_UNSUPPORTED", "sat-IN, mni-IN"):
        run(fake, tmp_path, audio, language="sat-IN")
    assert fake.jobs["job-1"]["model"] == "saaras:v3" and fake.created() == 1  # decided up front: no failed v4 request first


def test_a_language_v4_refuses_is_retried_once_on_v3(one_part, audio, tmp_path):
    fake = FakeSarvam(fail={("POST", "/speech-to-text/job/v1"): lambda body: 422 if body["job_parameters"]["model"].endswith("v4") else 0})
    with patch.object(settings, "SARVAM_MODEL", "saaras:v4"):
        run(fake, tmp_path, audio, language="doi-IN")
    assert [c[4]["job_parameters"]["model"] for c in fake.calls if c[2] == "/speech-to-text/job/v1"] == ["saaras:v4", "saaras:v3"]
    assert fake.jobs["job-1"]["model"] == "saaras:v3"


def test_a_rejected_key_is_not_mistaken_for_an_unsupported_language(one_part, audio, tmp_path):
    fake = FakeSarvam(fail={("POST", "/speech-to-text/job/v1"): 401})
    with patch.object(settings, "SARVAM_MODEL", "saaras:v4"), pytest.raises(SarvamRejected):
        run(fake, tmp_path, audio, language="hi-IN")
    assert fake.created() == 1  # no fallback attempt for 401 / 403 / 5xx


def test_v3_configured_means_no_fallback_logic_at_all(one_part, audio, tmp_path):
    fake = FakeSarvam(fail={("POST", "/speech-to-text/job/v1"): 422})
    with pytest.raises(SarvamRejected):
        run(fake, tmp_path, audio, language="hi-IN")
    assert fake.created() == 1


# ---------------------------------------------------------------- the text APIs
def test_split_for_limit_cuts_at_sentence_ends_and_never_loses_text():
    text = ("First sentence here. " * 30).strip()
    pieces = split_for_limit(text, 100)
    assert all(len(p) <= 100 for p in pieces) and len(pieces) > 1
    assert all(p.endswith(".") for p in pieces)  # cut after sentences
    assert " ".join(pieces) == text
    assert split_for_limit("short", 100) == ["short"] and split_for_limit("   ", 100) == []
    assert "".join(split_for_limit("x" * 250, 100)) == "x" * 250  # no spaces at all: hard cut, still lossless
    hindi = "यह एक वाक्य है। " * 40
    assert " ".join(split_for_limit(hindi, 120)).split() == hindi.split()


def test_text_plan_follows_the_documented_language_support():
    assert text_plan("hi-IN") == {"roman": ("hi-IN", None), "en": ("hi-IN", "sarvam-translate:v1")}
    assert text_plan("en-IN") == {"roman": None, "en": None}  # already English and Latin
    assert text_plan("sat-IN") == {"roman": None, "en": ("sat-IN", "sarvam-translate:v1")}  # translate knows Santali, transliterate does not
    assert text_plan(None) == {"roman": ("auto", None), "en": ("auto", "mayura:v1")}  # only mayura accepts "auto"
    assert text_plan("unknown") == text_plan(None) and text_plan("hi") == text_plan(None)  # not a Sarvam code
    with patch.object(settings, "SARVAM_TRANSLATE_MODEL", "mayura:v1"):
        assert text_plan("ur-IN")["en"] is None and text_plan("ta-IN")["en"] == ("ta-IN", "mayura:v1")  # mayura has 11 languages only


def spans(*texts):
    return [ChunkSpan(i * 25.0, i * 25.0 + 30, t) for i, t in enumerate(texts)]


def test_text_variants_are_made_from_each_chunks_native_text(tmp_path):
    fake = FakeSarvam()
    sp = spans("नमस्ते दोस्तों", "आज हम बात करेंगे")
    out = add_text_variants(sp, "hi-IN", client=client_for(fake))
    assert [s.text_roman for s in sp] == ["R:नमस्ते दोस्तों", "R:आज हम बात करेंगे"]
    assert [s.text_en for s in sp] == ["EN:नमस्ते दोस्तों", "EN:आज हम बात करेंगे"]
    assert [s.text for s in sp] == ["नमस्ते दोस्तों", "आज हम बात करेंगे"]  # the native text is untouched
    paths = sorted(p for p, _ in fake.text_calls)
    assert paths == ["/translate", "/translate", "/transliterate", "/transliterate"]
    bodies = {p: b for p, b in fake.text_calls}
    assert bodies["/translate"]["target_language_code"] == "en-IN" and bodies["/translate"]["source_language_code"] == "hi-IN"
    assert bodies["/translate"]["model"] == "sarvam-translate:v1"
    assert bodies["/transliterate"]["target_language_code"] == "en-IN" and "model" not in bodies["/transliterate"]
    assert out["chars"] == {"transliterate": 14 + 16, "translate": 14 + 16}  # exactly the characters sent: that is what is billed
    assert all(c[3] == "sarvam-secret" for c in fake.calls)  # the key goes with every text call


def test_english_speech_needs_no_text_api_calls_at_all():
    fake = FakeSarvam()
    sp = spans("hello there")
    out = add_text_variants(sp, "en-IN", client=client_for(fake))
    assert fake.text_calls == [] and sp[0].text_roman is None and sp[0].text_en is None
    assert out["chars"] == {"transliterate": 0, "translate": 0} and out["skipped"] == ["roman", "en"]


def test_unsupported_language_skips_only_the_variant_the_api_cannot_do():
    fake = FakeSarvam()
    sp = spans("ᱥᱟᱱᱛᱟᱲᱤ")
    out = add_text_variants(sp, "sat-IN", client=client_for(fake))
    assert sp[0].text_roman is None and sp[0].text_en == "EN:ᱥᱟᱱᱛᱟᱲᱤ" and out["skipped"] == ["roman"]


def test_the_variants_setting_chooses_which_text_apis_are_called():
    for setting, expected in (("roman", ["/transliterate"]), ("en", ["/translate"]), ("", []), ("en,roman", ["/transliterate", "/translate"])):
        fake = FakeSarvam()
        with patch.object(settings, "SARVAM_TEXT_VARIANTS", setting):
            add_text_variants(spans("नमस्ते"), "hi-IN", client=client_for(fake))
        assert sorted({p for p, _ in fake.text_calls}) == sorted(expected)
    with patch.object(settings, "SARVAM_TEXT_VARIANTS", "roman,klingon"), pytest.raises(ValueError, match="klingon"):
        settings.sarvam_text_variants


def test_texts_over_the_request_limit_are_split_and_joined_again():
    fake = FakeSarvam()
    sentence = "यह एक वाक्य है। "
    long = (sentence * 150).strip()  # about 2,400 characters: over the 2000 (translate) and 1000 (transliterate) limits
    sp = spans(long)
    add_text_variants(sp, "hi-IN", client=client_for(fake))
    translate = [b["input"] for p, b in fake.text_calls if p == "/translate"]
    translit = [b["input"] for p, b in fake.text_calls if p == "/transliterate"]
    assert len(translate) == 2 and all(len(t) <= 2000 for t in translate)
    assert len(translit) == 3 and all(len(t) <= 1000 for t in translit)
    assert sp[0].text_en.replace("EN:", "").split() == long.split()  # nothing lost in the middle
    assert sp[0].text_roman.replace("R:", "").split() == long.split()


def test_mayura_has_a_smaller_limit_than_sarvam_translate():
    fake = FakeSarvam()
    with patch.object(settings, "SARVAM_TRANSLATE_MODEL", "mayura:v1"), patch.object(settings, "SARVAM_TEXT_VARIANTS", "en"):
        add_text_variants(spans(("यह एक वाक्य है। " * 100).strip()), "hi-IN", client=client_for(fake))
    assert all(len(b["input"]) <= 1000 and b["model"] == "mayura:v1" for _, b in fake.text_calls)


def test_a_text_the_api_refuses_leaves_only_that_value_empty():
    fake = FakeSarvam(fail={("POST", "/translate"): lambda body: 422 if "bad" in body["input"] else 0})
    sp = spans("good one", "bad one")
    add_text_variants(sp, "hi-IN", client=client_for(fake), variants=["en"])
    assert sp[0].text_en == "EN:good one" and sp[1].text_en is None


def test_a_rejected_key_or_no_credits_fails_the_whole_step():
    fake = FakeSarvam(fail={("POST", "/translate"): 403})
    with pytest.raises(SarvamRejected):
        add_text_variants(spans("one", "two"), "hi-IN", client=client_for(fake))


def test_an_outage_of_the_text_api_is_retryable():
    fake = FakeSarvam(fail={("POST", "/transliterate"): 503})
    with pytest.raises(SarvamUnavailable):
        add_text_variants(spans("one"), "hi-IN", client=client_for(fake))


def test_text_api_responses_without_the_text_are_an_error_not_silence():
    c = client_for(SimpleNamespace(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))))
    with pytest.raises(SarvamRejected):
        c.translate("x", "hi-IN", "sarvam-translate:v1")
    with pytest.raises(SarvamRejected):
        c.transliterate("x", "hi-IN")
