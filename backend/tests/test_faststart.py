"""MP4 faststart: detection of the index position, the ffmpeg stream-copy command, the ingest step and the remux script."""
import struct
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.scripts import remux_faststart
from app.services import media, storage
from app.workers import tasks


def box(kind: bytes, payload: bytes = b"", big: bool = False) -> bytes:
    if big:  # 64-bit "largesize" form
        return struct.pack(">I", 1) + kind + struct.pack(">Q", 16 + len(payload)) + payload
    return struct.pack(">I", 8 + len(payload)) + kind + payload


FTYP = box(b"ftyp", b"isom\x00\x00\x02\x00")
SLOW = FTYP + box(b"mdat", b"x" * 100) + box(b"moov", b"m" * 20)  # index at the END: what a camera or screen recorder writes
FAST = FTYP + box(b"moov", b"m" * 20) + box(b"mdat", b"x" * 100)


def write(tmp_path, data: bytes, name="v.mp4") -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# ---------------------------------------------------------------- detection
def test_moov_position_is_detected(tmp_path):
    assert media.moov_before_mdat(write(tmp_path, SLOW)) is False
    assert media.moov_before_mdat(write(tmp_path, FAST)) is True


def test_boxes_with_64_bit_sizes_and_free_atoms_are_walked(tmp_path):
    data = FTYP + box(b"free", b"\x00" * 30) + box(b"mdat", b"x" * 50, big=True) + box(b"moov", b"m")
    assert media.moov_before_mdat(write(tmp_path, data)) is False
    data = FTYP + box(b"free", b"\x00" * 30) + box(b"moov", b"m", big=True) + box(b"mdat", b"x")
    assert media.moov_before_mdat(write(tmp_path, data)) is True


def test_a_box_that_runs_to_the_end_of_the_file(tmp_path):
    data = FTYP + struct.pack(">I", 0) + b"mdat" + b"x" * 40  # size 0 = until EOF
    assert media.moov_before_mdat(write(tmp_path, data)) is False


def test_non_mp4_and_garbage_are_unknown_not_a_crash(tmp_path):
    assert media.moov_before_mdat(write(tmp_path, b"\x1a\x45\xdf\xa3" + b"\x00" * 50)) is None  # a matroska header
    assert media.moov_before_mdat(write(tmp_path, b"")) is None
    assert media.moov_before_mdat(write(tmp_path, b"abc")) is None
    assert media.moov_before_mdat(write(tmp_path, struct.pack(">I", 3) + b"free")) is None  # impossible size: stop


def test_needs_faststart_only_for_mp4_family_with_the_index_last(tmp_path):
    assert media.needs_faststart(write(tmp_path, SLOW, "a.mp4")) is True
    assert media.needs_faststart(write(tmp_path, SLOW, "a.mov")) is True
    assert media.needs_faststart(write(tmp_path, FAST, "b.mp4")) is False
    assert media.needs_faststart(write(tmp_path, SLOW, "c.webm")) is False  # no moov atom in webm/mkv
    assert media.needs_faststart(write(tmp_path, SLOW, "d.mkv")) is False


# ---------------------------------------------------------------- remux
def test_remux_is_a_stream_copy_with_faststart(tmp_path):
    out = tmp_path / "out.mp4"

    def fake_run(cmd, **kw):
        out.write_bytes(FAST)
        return SimpleNamespace(stdout="", stderr="")

    with patch.object(media.shutil, "which", return_value="/bin/ffmpeg"), \
         patch.object(media.subprocess, "run", side_effect=fake_run) as run:
        assert media.remux_faststart("in.mp4", str(out)) == str(out)
    cmd = run.call_args[0][0]
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-c") + 1] == "copy"  # no re-encode
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert "-map" in cmd and "0:v" in cmd and "0:a?" in cmd  # keeps all video and audio streams


def test_remux_that_did_not_move_the_index_is_an_error(tmp_path):
    out = tmp_path / "out.mp4"
    with patch.object(media.shutil, "which", return_value="/bin/ffmpeg"), \
         patch.object(media.subprocess, "run", side_effect=lambda *a, **k: out.write_bytes(SLOW)):
        with pytest.raises(media.MediaError, match="not at the start"):
            media.remux_faststart("in.mp4", str(out))


# ---------------------------------------------------------------- ingest step
@pytest.fixture
def video():
    return SimpleNamespace(id=uuid.uuid4(), storage_key="raw/abc.mp4")


def test_ingest_replaces_the_stored_object_with_the_faststart_copy(tmp_path, video):
    out = tmp_path / "faststart.mp4"
    out.write_bytes(FAST)
    with patch.object(media, "needs_faststart", return_value=True), \
         patch.object(media, "remux_faststart", return_value=str(out)) as remux, \
         patch.object(storage, "upload_fileobj") as upload:
        tasks._store_faststart(video, str(tmp_path / "abc.mp4"), str(tmp_path), "job")
    remux.assert_called_once()
    fileobj, key, content_type = upload.call_args[0]
    assert key == "raw/abc.mp4" and content_type == "video/mp4"  # same key: in place, the DB row is unchanged


def test_ingest_leaves_a_file_that_is_already_faststart_alone(tmp_path, video):
    with patch.object(media, "needs_faststart", return_value=False), patch.object(media, "remux_faststart") as remux, \
         patch.object(storage, "upload_fileobj") as upload:
        tasks._store_faststart(video, str(tmp_path / "abc.mp4"), str(tmp_path), "job")
    remux.assert_not_called()
    upload.assert_not_called()


@pytest.mark.parametrize("failing", ["remux", "upload"])
def test_a_faststart_problem_never_fails_the_job(tmp_path, video, failing, caplog):
    out = tmp_path / "faststart.mp4"
    out.write_bytes(FAST)
    remux = MagicMock(side_effect=media.MediaError("boom", raw="ffmpeg said no")) if failing == "remux" else \
        MagicMock(return_value=str(out))
    upload = MagicMock(side_effect=ConnectionError("s3 down")) if failing == "upload" else MagicMock()
    with patch.object(media, "needs_faststart", return_value=True), patch.object(media, "remux_faststart", remux), \
         patch.object(storage, "upload_fileobj", upload), caplog.at_level("WARNING"):
        tasks._store_faststart(video, str(tmp_path / "abc.mp4"), str(tmp_path), "job-1")  # must not raise
    assert any("faststart remux skipped" in m and "job-1" in m for m in caplog.messages)


# ---------------------------------------------------------------- the script
def make_downloader(data: bytes):
    return lambda key, path: open(path, "wb").write(data)


@pytest.fixture(autouse=True)
def _head_of_slow_file_by_default():
    """read_head is a ranged S3 GET: the tests give it the first bytes of whatever the 'object' holds."""
    with patch.object(storage, "read_head", return_value=SLOW[:64]) as head:
        yield head


def test_script_fixes_a_slow_video_verifies_and_uploads_in_place(tmp_path):
    uploads = []
    fast = FAST

    def fake_remux(src, dst):
        open(dst, "wb").write(fast)
        return dst

    with patch.object(storage, "download_to", side_effect=make_downloader(SLOW)), \
         patch.object(media, "remux_faststart", side_effect=fake_remux), \
         patch.object(media, "duration_sec", return_value=60), patch.object(media, "has_audio", return_value=True), \
         patch.object(storage, "upload_fileobj", side_effect=lambda f, k, c: uploads.append((k, c))):
        assert remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=True) == "fixed"
    assert uploads == [("originals/raw/a.mp4", "video/mp4"), ("raw/a.mp4", "video/mp4")]  # backup first, then replace


def test_script_dry_run_touches_nothing(tmp_path):
    with patch.object(storage, "download_to", side_effect=make_downloader(SLOW)), \
         patch.object(media, "remux_faststart") as remux, patch.object(storage, "upload_fileobj") as upload:
        assert remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=True, keep_original=False) == "would-fix"
    remux.assert_not_called()
    upload.assert_not_called()


def test_script_skips_fast_and_non_mp4_files(tmp_path):
    with patch.object(storage, "download_to", side_effect=make_downloader(FAST)) as dl, \
         patch.object(storage, "upload_fileobj") as upload:
        assert remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=False) == "already-fast"
        assert remux_faststart.process("raw/b.webm", str(tmp_path), dry_run=False, keep_original=False).startswith("skipped")
    assert dl.call_count == 1  # the webm was never even downloaded
    upload.assert_not_called()


def test_script_does_not_replace_the_original_when_verification_fails(tmp_path):
    def fake_remux(src, dst):
        open(dst, "wb").write(FAST)
        return dst

    durations = iter([30, 60])  # remuxed (30) vs original (60)
    with patch.object(storage, "download_to", side_effect=make_downloader(SLOW)), \
         patch.object(media, "remux_faststart", side_effect=fake_remux), \
         patch.object(media, "duration_sec", side_effect=lambda p: next(durations)), \
         patch.object(storage, "upload_fileobj") as upload:
        with pytest.raises(media.MediaError, match="duration"):
            remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=False)
    upload.assert_not_called()


# ---------------------------------------------------------------- head check: nothing is downloaded for fast files
def test_head_decides_that_a_fast_file_needs_no_download(tmp_path, _head_of_slow_file_by_default):
    _head_of_slow_file_by_default.return_value = FAST[:64]
    with patch.object(storage, "download_to") as download:
        assert remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=False) == "already-fast"
    download.assert_not_called()


def test_head_walk_is_only_trusted_when_it_finds_the_index_first():
    assert media.moov_before_mdat_in_head(FAST) is True
    assert media.moov_before_mdat_in_head(FAST[:20]) is not True  # cut before the index header: undecided, never a false "fast"
    assert media.moov_before_mdat_in_head(SLOW) is False
    assert media.moov_before_mdat_in_head(b"") is None


def test_a_slow_file_is_still_downloaded_and_fixed_after_the_head_check(tmp_path):
    def fake_remux(src, dst):
        open(dst, "wb").write(FAST)
        return dst

    with patch.object(storage, "download_to", side_effect=make_downloader(SLOW)) as download,          patch.object(media, "remux_faststart", side_effect=fake_remux),          patch.object(media, "duration_sec", return_value=60), patch.object(media, "has_audio", return_value=True),          patch.object(storage, "upload_fileobj"):
        assert remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=False) == "fixed"
    download.assert_called_once()


def test_running_it_twice_changes_nothing_the_second_time(tmp_path):
    """The stored object after the first run reads as faststart, so the second run downloads and uploads nothing."""
    state = {"data": SLOW}

    def fake_remux(src, dst):
        open(dst, "wb").write(FAST)
        return dst

    def head(key, n=65536):
        return state["data"][:n]

    def upload(f, key, content_type):
        state["data"] = f.read()

    with patch.object(storage, "read_head", side_effect=head),          patch.object(storage, "download_to", side_effect=lambda k, p: open(p, "wb").write(state["data"])) as download,          patch.object(media, "remux_faststart", side_effect=fake_remux),          patch.object(media, "duration_sec", return_value=60), patch.object(media, "has_audio", return_value=True),          patch.object(storage, "upload_fileobj", side_effect=upload) as up:
        first = remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=False)
        downloads_after_first, uploads_after_first = download.call_count, up.call_count
        second = remux_faststart.process("raw/a.mp4", str(tmp_path), dry_run=False, keep_original=False)
    assert (first, second) == ("fixed", "already-fast")
    assert download.call_count == downloads_after_first and up.call_count == uploads_after_first


# ---------------------------------------------------------------- main(): job skip, failures, exit code
def run_main(videos, busy, argv, process_result):
    db = MagicMock()
    db.__enter__.return_value = db
    db.execute.return_value.all.return_value = videos
    db.scalars.return_value.all.return_value = list(busy)
    with patch.object(remux_faststart, "SessionLocal", return_value=db),          patch.object(remux_faststart, "process", side_effect=process_result) as process:
        code = remux_faststart.main(argv)
    return code, process


def test_main_skips_videos_that_are_being_processed():
    a, b = uuid.uuid4(), uuid.uuid4()
    code, process = run_main([(a, "A", "raw/a.mp4"), (b, "B", "raw/b.mp4")], {a}, [], lambda *x, **k: "fixed")
    assert code == 0 and process.call_count == 1 and process.call_args[0][0] == "raw/b.mp4"


def test_main_continues_after_a_failure_and_exits_1():
    a, b = uuid.uuid4(), uuid.uuid4()

    def process(key, tmp, **kw):
        if key == "raw/a.mp4":
            raise media.MediaError("broken file")
        return "fixed"

    code, p = run_main([(a, "A", "raw/a.mp4"), (b, "B", "raw/b.mp4")], set(), ["--dry-run"], process)
    assert code == 1 and p.call_count == 2 and p.call_args[1]["dry_run"] is True
