"""Hosted transcription with Sarvam (Saaras): ONE batch speech-to-text job per video, then Sarvam's text APIs for the other scripts.

Why batch: the real-time REST endpoint takes at most 30 seconds of audio per request; batch takes up to 2 hours per file.
Flow (verified against the official SDK source, sarvamai 0.1.37, and docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api):

  POST /speech-to-text/job/v1                 {"job_parameters": {"model", "mode": "transcribe", "language_code"}}  -> job_id
  POST /speech-to-text/job/v1/upload-files    {"job_id", "files": [names]}  -> upload_urls[name].file_url   (then PUT each file there,
                                              header x-ms-blob-type: BlockBlob)
  POST /speech-to-text/job/v1/{job_id}/start
  GET  /speech-to-text/job/v1/{job_id}/status -> job_state Accepted|Pending|Running|Completed|PartiallyCompleted|Failed, job_details[]
  POST /speech-to-text/job/v1/download-files  {"job_id", "files": [output names]} -> download_urls[name].file_url   (then GET each)
  header on every call to api.sarvam.ai: api-subscription-key

Each output file is JSON: {"transcript", "language_code", "timestamps": {"chunks": [...], "start_time_seconds": [...],
"end_time_seconds": [...]}}: sentence/phrase-level timestamps (word-level ones do not exist).

The transcript is fetched once, in the speaker's own script (mode "transcribe", billed for the audio once). The Latin-script and
English versions of each chunk are made from the NATIVE chunk text with the text APIs (docs read 2026-10-20):

  POST /transliterate {"input", "source_language_code", "target_language_code": "en-IN"} -> {"transliterated_text"}   (Indic -> Latin script)
  POST /translate     {"input", "source_language_code", "target_language_code": "en-IN", "model"} -> {"translated_text"}
  both billed INR 20 per 10K characters of input; translate takes 2000 characters per request (sarvam-translate:v1) or 1000 (mayura:v1).

Idempotency: the job id and its status are handed to `on_job` right after each step, and a retry passes them back in, so a failure after
the job was created resumes the same job (re-uploading or re-starting it only if that step had not happened) instead of paying twice."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import httpx

from app.core.config import settings
from app.services import languages, media
from app.services.transcription import NoSpeechError

log = logging.getLogger(__name__)
DONE_STATES = {"completed", "partiallycompleted"}
# videos.sarvam_job_status values we write ourselves (Sarvam's own states are stored lower-case as well)
CREATED, UPLOADED, STARTED, CONSUMED = "created", "uploaded", "started", "consumed"
REUSABLE = {CREATED, UPLOADED, STARTED, "accepted", "pending", "running", "completed", "partiallycompleted"}  # not failed, not consumed

TRANSLATE_LIMITS = {"sarvam-translate:v1": 2000, "mayura:v1": 1000}
TEXT_RETRIES = 2


# ---- errors: user_message is short and path-free (it is stored in jobs.error); the details go to the log
class SarvamError(Exception):
    user_message = "Transcription with Sarvam failed. Please try again or contact an administrator."


class SarvamNotConfigured(SarvamError):
    user_message = "Transcription is not set up: an administrator needs to configure SARVAM_API_KEY."


class SarvamUnavailable(SarvamError):
    """Network trouble, timeouts, 429 or 5xx: worth retrying later."""
    user_message = "The transcription service was unreachable. Please try again in a moment."


class SarvamRejected(SarvamError):
    """A 4xx: bad key, no credits, unsupported file."""
    user_message = "The transcription service rejected the request. An administrator should check SARVAM_API_KEY and the account credits."

    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        self.status = status


class SarvamJobFailed(SarvamError):
    user_message = "The transcription service could not process this audio."


class SarvamNoTimestamps(SarvamError):
    """The reply had text but no segment timestamps. Without them a moment cannot be found, so ingestion must not go on."""
    user_message = "The transcription service returned no timestamps for this video."


@dataclass
class TranscriptResult:
    segments: list[dict]  # [{start, end, text}] in seconds, native language and script
    language: str
    engine: str
    audio_seconds: float
    billed_seconds: float = 0.0  # audio seconds the provider charges for (one job: the audio duration)
    extra: dict = field(default_factory=dict)  # job_id, model


# ---- pure helpers
def parse_output(payload: dict, offset: float = 0.0) -> list[dict]:
    """Segments [{start, end, text}] from one output file. Empty list for silence. SarvamNoTimestamps if there is text but the
    timestamp arrays are missing or inconsistent."""
    ts = payload.get("timestamps") or {}
    chunks, starts, ends = ts.get("chunks"), ts.get("start_time_seconds"), ts.get("end_time_seconds")
    ok = isinstance(chunks, list) and isinstance(starts, list) and isinstance(ends, list) and len(chunks) == len(starts) == len(ends)
    if not ok:
        if (payload.get("transcript") or "").strip():
            raise SarvamNoTimestamps("output has a transcript but no usable timestamps.chunks/start_time_seconds/end_time_seconds")
        return []
    return [{"start": float(s) + offset, "end": float(e) + offset, "text": str(c).strip()}
            for c, s, e in zip(chunks, starts, ends) if str(c).strip()]


def split_for_limit(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` characters, cut at sentence ends when possible, else at spaces, else hard. Joining the pieces with a
    space gives the text back (modulo whitespace)."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = max(window.rfind(m) for m in (". ", "? ", "! ", "। ", "\n")) + 1  # after the punctuation mark
        if cut <= limit // 4:
            cut = window.rfind(" ") + 1
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return [p for p in pieces if p]


def v4_supported(language_code: str) -> bool:
    return language_code not in settings.sarvam_v4_unsupported


def text_plan(language: str | None) -> dict:
    """For a detected/chosen language: {"roman": (source_code, None) | None, "en": (source_code, model) | None}. None = that variant is
    not made: the text is already English / Latin, or the text API does not support the language."""
    code = language if language in languages.CODES else None  # "unknown", Whisper's "hi" and garbage all mean: let Sarvam detect
    if code == "en-IN":
        return {"roman": None, "en": None}
    plan: dict = {"roman": None, "en": None}
    if code is None or code in languages.TRANSLIT_SOURCES:
        plan["roman"] = (code or "auto", None)
    model = settings.SARVAM_TRANSLATE_MODEL
    if code is None:  # only mayura accepts "auto"
        plan["en"] = ("auto", "mayura:v1")
    elif code in (languages.SARVAM_TRANSLATE_SOURCES if model == "sarvam-translate:v1" else languages.MAYURA_SOURCES):
        plan["en"] = (code, model)
    return plan


# ---- HTTP client
class SarvamClient:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.api_key = settings.SARVAM_API_KEY if api_key is None else api_key
        self.base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self.model = model or settings.SARVAM_MODEL
        self._client = client
        self._sleep, self._clock = sleep, clock
        self.text_chars = {"transliterate": 0, "translate": 0}  # characters sent to the text APIs (what they bill), for the cost log

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def _call(self, method: str, path: str, *, json: dict | None = None, retries: int = 2, timeout: float = 60) -> dict:
        if not self.api_key:
            raise SarvamNotConfigured("SARVAM_API_KEY is empty")
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self._http().request(method, f"{self.base_url}/{path.lstrip('/')}", json=json, timeout=timeout,
                                         headers={"api-subscription-key": self.api_key})
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
            else:
                if r.status_code < 400:
                    return r.json()
                if not (r.status_code == 429 or r.status_code >= 500):
                    log.warning("sarvam %s %s -> %d", method, path, r.status_code)
                    raise SarvamRejected(f"{method} {path} answered {r.status_code}", r.status_code)
                last = SarvamUnavailable(f"{method} {path} answered {r.status_code}")
            if attempt < retries:
                self._sleep(min(10.0, 1.0 * 2 ** attempt))
        raise SarvamUnavailable(f"{method} {path}: {type(last).__name__}") from last

    # -- speech to text (batch)
    def create_job(self, language_code: str) -> tuple[str, str]:
        """Starts ONE transcription job (mode "transcribe", own script). Returns (job_id, model used). With saaras:v4 configured, a
        language listed in SARVAM_V4_UNSUPPORTED, or a request the API refuses with 400/422, is retried once on the fallback model."""
        model = self.model
        if model.endswith(":v4") and not v4_supported(language_code):
            log.info("sarvam: %s is not supported by %s, using %s", language_code, model, settings.SARVAM_FALLBACK_MODEL)
            model = settings.SARVAM_FALLBACK_MODEL
        try:
            return self._create(model, language_code), model
        except SarvamRejected as exc:
            if not model.endswith(":v4") or exc.status not in (400, 422) or settings.SARVAM_FALLBACK_MODEL == model:
                raise
            log.warning("sarvam: %s refused language %s (HTTP %s), retrying with %s", model, language_code, exc.status,
                        settings.SARVAM_FALLBACK_MODEL)
            model = settings.SARVAM_FALLBACK_MODEL
            return self._create(model, language_code), model

    def _create(self, model: str, language_code: str) -> str:
        data = self._call("POST", "speech-to-text/job/v1",
                          json={"job_parameters": {"model": model, "mode": "transcribe", "language_code": language_code}})
        try:
            return data["job_id"]
        except (KeyError, TypeError) as exc:
            raise SarvamRejected("job creation returned no job_id") from exc

    def upload(self, job_id: str, files: list[tuple[str, str]]) -> None:
        """files = [(name, local path)]. The API hands out pre-signed upload URLs; each file is PUT to its own."""
        data = self._call("POST", "speech-to-text/job/v1/upload-files", json={"job_id": job_id, "files": [n for n, _ in files]})
        urls = data.get("upload_urls") or {}
        for name, path in files:
            try:
                url = urls[name]["file_url"]
            except (KeyError, TypeError) as exc:
                raise SarvamRejected(f"no upload URL for {name}") from exc
            with open(path, "rb") as f:
                try:
                    r = self._http().put(url, content=f.read(), timeout=600,
                                         headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "audio/mpeg"})
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    raise SarvamUnavailable(f"upload of {name}: {type(exc).__name__}") from exc
            if r.status_code >= 400:
                raise SarvamUnavailable(f"upload of {name} answered {r.status_code}") if r.status_code >= 500 else SarvamRejected(
                    f"upload of {name} answered {r.status_code}", r.status_code)

    def start(self, job_id: str) -> None:
        self._call("POST", f"speech-to-text/job/v1/{job_id}/start")

    def status(self, job_id: str) -> dict:
        # Tolerant on purpose (about 35 s of backoff): a failure here makes the worker retry the video, and although a retry now
        # resumes the same job, there is no reason to burn a retry on a hiccup.
        return self._call("GET", f"speech-to-text/job/v1/{job_id}/status", retries=6)

    def wait(self, job_id: str, *, on_poll: Callable[[str], None] | None = None) -> dict:
        """Polls until the job is finished. SarvamJobFailed if it failed; SarvamUnavailable if it takes longer than
        SARVAM_JOB_TIMEOUT_SEC (the worker retries and resumes polling the same job)."""
        deadline = self._clock() + settings.SARVAM_JOB_TIMEOUT_SEC
        while True:
            st = self.status(job_id)
            state = str(st.get("job_state", "")).lower()
            if on_poll:
                on_poll(state)
            if state in DONE_STATES:
                return st
            if state == "failed":
                log.warning("sarvam job %s failed: %s", job_id, st.get("error_message"))
                raise SarvamJobFailed(f"job {job_id} failed: {st.get('error_message')}")
            if self._clock() > deadline:
                raise SarvamUnavailable(f"job {job_id} still {state or 'unknown'} after {settings.SARVAM_JOB_TIMEOUT_SEC}s")
            self._sleep(settings.SARVAM_POLL_SEC)

    def results(self, job_id: str, status: dict, input_names: list[str]) -> list[dict]:
        """The parsed output JSON for each input file, in the order of `input_names`."""
        out_for: dict[str, str] = {}
        for d in status.get("job_details") or []:
            if str(d.get("state", "")).lower() not in ("success", "completed"):
                raise SarvamJobFailed(f"file failed: {d.get('error_message') or d.get('exception_name')}")
            ins, outs = d.get("inputs") or [], d.get("outputs") or []
            if ins and outs:
                out_for[ins[0]["file_name"]] = outs[0]["file_name"]
        missing = [n for n in input_names if n not in out_for]
        if missing:
            raise SarvamJobFailed(f"no output for {missing}")
        data = self._call("POST", "speech-to-text/job/v1/download-files", json={"job_id": job_id, "files": [out_for[n] for n in input_names]})
        urls = data.get("download_urls") or {}
        payloads = []
        for n in input_names:
            try:
                url = urls[out_for[n]]["file_url"]
            except (KeyError, TypeError) as exc:
                raise SarvamRejected(f"no download URL for {out_for[n]}") from exc
            try:
                r = self._http().get(url, timeout=120)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise SarvamUnavailable(f"download of {out_for[n]}: {type(exc).__name__}") from exc
            if r.status_code >= 400:
                raise SarvamUnavailable(f"download answered {r.status_code}")
            payloads.append(r.json())
        return payloads

    # -- text APIs (billed per character of input)
    def transliterate(self, text: str, source: str) -> str:
        """Indic script -> Latin script. `source` is a language code or "auto"."""
        self.text_chars["transliterate"] += len(text)
        data = self._call("POST", "transliterate", retries=TEXT_RETRIES, timeout=settings.SARVAM_TEXT_TIMEOUT_SEC,
                          json={"input": text, "source_language_code": source, "target_language_code": "en-IN"})
        try:
            return str(data["transliterated_text"]).strip()
        except (KeyError, TypeError) as exc:
            raise SarvamRejected("transliterate returned no transliterated_text") from exc

    def translate(self, text: str, source: str, model: str) -> str:
        """Any supported language -> English. `source` is a language code, or "auto" with mayura:v1."""
        self.text_chars["translate"] += len(text)
        data = self._call("POST", "translate", retries=TEXT_RETRIES, timeout=settings.SARVAM_TEXT_TIMEOUT_SEC,
                          json={"input": text, "source_language_code": source, "target_language_code": "en-IN", "model": model})
        try:
            return str(data["translated_text"]).strip()
        except (KeyError, TypeError) as exc:
            raise SarvamRejected("translate returned no translated_text") from exc


# ---- the transcription of one video's audio: one job, resumable
def _open_job(client: SarvamClient, files: list[tuple[str, str]], code: str, total: float, job_id: str | None, job_status: str | None,
              on_job: Callable[[str, str], None]) -> str:
    """The id of a job that is running or finished for this audio: the saved one if it is still usable, else a new one. Every step is
    reported to `on_job` straight after it happened, so a crash between two steps leaves the truth on record."""
    if job_id and (job_status or "").lower() in REUSABLE:
        try:
            live = str(client.status(job_id).get("job_state", "")).lower()
        except SarvamRejected as exc:  # Sarvam does not know the job any more (expired / wrong account)
            log.warning("sarvam: saved job %s cannot be read (HTTP %s), starting a new one", job_id, exc.status)
            live = "failed"
        if live == "failed":
            log.warning("sarvam: saved job %s failed on Sarvam's side, starting a new one", job_id)
        else:
            saved = (job_status or "").lower()
            log.info("sarvam: resuming job %s (saved status %s, Sarvam says %s): no new job is created", job_id, saved, live or "?")
            if live in ("", "accepted"):  # nothing runs yet: finish the steps the crashed attempt did not
                if saved == CREATED:
                    client.upload(job_id, files)
                    on_job(job_id, UPLOADED)
                    saved = UPLOADED
                if saved == UPLOADED:
                    client.start(job_id)
                    on_job(job_id, STARTED)
            return job_id
    job_id, model = client.create_job(code)
    on_job(job_id, CREATED)
    log.info("sarvam: created %s job %s (%d file(s), %.0f s audio)", model, job_id, len(files), total)
    client.upload(job_id, files)
    on_job(job_id, UPLOADED)
    client.start(job_id)
    on_job(job_id, STARTED)
    return job_id


def transcribe_audio(audio_path: str, tmp_dir: str, *, language: str | None = None,
                     on_progress: Callable[[float, float], None] | None = None, client: SarvamClient | None = None,
                     job_id: str | None = None, job_status: str | None = None,
                     on_job: Callable[[str, str], None] | None = None) -> TranscriptResult:
    """Splits the audio if it is over the batch limit, runs ONE batch job (native script), downloads and parses it. Pass the saved
    `job_id` / `job_status` of an earlier attempt to resume it instead of starting another. NoSpeechError if the audio has no speech."""
    client = client or SarvamClient()
    on_job = on_job or (lambda job_id, status: None)
    total = media.duration_float(audio_path)
    parts = media.split_audio(audio_path, tmp_dir, settings.SARVAM_MAX_FILE_SEC)
    names = [f"part{i:03d}.mp3" for i in range(len(parts))]
    files = [(n, p) for n, (p, _) in zip(names, parts)]
    report = on_progress or (lambda done, total: None)

    job = _open_job(client, files, languages.sarvam_code(language), total, job_id, job_status, on_job)
    report(0.1 * total, total)
    last = [None]

    def polled(state: str) -> None:
        if state and state != last[0]:  # one DB write per state change, not one per poll
            last[0] = state
            on_job(job, state)

    status = client.wait(job, on_poll=polled)
    payloads = client.results(job, status, names)
    segments: list[dict] = []
    detected = None
    for payload, (_, offset) in zip(payloads, parts):
        segments += parse_output(payload, offset)
        detected = detected or payload.get("language_code")
    report(total, total)
    if not segments:
        raise NoSpeechError("Sarvam returned no speech segments")
    lang = detected if detected and detected != "unknown" else language or "unknown"
    return TranscriptResult(segments=segments, language=lang, engine="sarvam", audio_seconds=total, billed_seconds=total,
                            extra={"job_id": job})


# ---- the other scripts, made from the native chunk text
def add_text_variants(spans: list, language: str | None, *, client: SarvamClient | None = None, variants: list[str] | None = None,
                      workers: int = 4) -> dict:
    """Sets `text_roman` / `text_en` on each chunk span from its native text, with /transliterate and /translate. Texts over the
    per-request limit are split at sentence ends and the pieces joined again. A variant the language does not allow (see text_plan)
    is skipped and logged. A refused single text (422) leaves just that value empty; any other refusal (bad key, no credits) or an
    outage raises. Returns {"chars": {"transliterate": n, "translate": n}, "skipped": [variant, ...]}."""
    client = client or SarvamClient()
    variants = settings.sarvam_text_variants if variants is None else variants
    plan = text_plan(language)
    skipped = [v for v in variants if plan.get(v) is None]
    for v in skipped:
        log.info("sarvam: no %s variant for language %s (already English/Latin, or not supported by the text API)", v, language)
    jobs = []  # (span, attribute, function that makes it)
    for span in spans:
        for v in variants:
            if plan.get(v) is None:
                continue
            source, model = plan[v]
            if v == "roman":
                limit, call = settings.SARVAM_TRANSLIT_MAX_CHARS, (lambda t, s=source: client.transliterate(t, s))
            else:
                limit, call = TRANSLATE_LIMITS.get(model, 1000), (lambda t, s=source, m=model: client.translate(t, s, m))
            jobs.append((span, "text_roman" if v == "roman" else "text_en", span.text, limit, call))

    def run(job):
        span, attr, text, limit, call = job
        try:
            out = " ".join(call(piece) for piece in split_for_limit(text, limit))
        except SarvamRejected as exc:
            if exc.status != 422:
                raise
            log.warning("sarvam: %s refused one text (HTTP 422), leaving %s empty for that chunk", attr, attr)
            return
        setattr(span, attr, out.strip() or None)

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(run, jobs))  # list(): re-raises the first failure
    return {"chars": dict(client.text_chars), "skipped": skipped}
