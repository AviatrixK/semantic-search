"""backfill_sha256: dry-run writes nothing, re-runs are safe, duplicates are reported not hashed, the untouched original wins."""
import hashlib
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import text

from app.models import Video
from app.scripts import backfill_sha256
from app.services import storage


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture
def library(api, db_session):
    """Four legacy videos (no hash). Objects: a and c hold the SAME bytes, b is different, d is unreadable."""
    ids = {n: uuid.uuid4() for n in "abcd"}
    for n, vid in ids.items():
        db_session.add(Video(id=vid, title=f"video {n}", storage_key=f"raw/{n}.mp4"))
    db_session.commit()
    objects = {"raw/a.mp4": "same", "raw/b.mp4": "other", "raw/c.mp4": "same"}

    def sha(key):
        if key not in objects:
            raise ConnectionError("S3 down")
        return h(objects[key])

    with patch.object(storage, "sha256_of_object", side_effect=sha) as sha_mock, \
         patch.object(storage, "object_exists", return_value=False) as exists:
        yield SimpleNs(ids=ids, objects=objects, sha=sha_mock, exists=exists, db=db_session)


class SimpleNs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def hashes(db):
    db.expire_all()
    return dict(db.execute(text("SELECT title, sha256 FROM videos")).all())


def test_dry_run_reads_everything_and_writes_nothing(library, capsys):
    assert backfill_sha256.main(["--dry-run"]) == 1  # video d cannot be read
    out = capsys.readouterr().out
    assert "would-set" in out and "dry run: nothing written" in out
    assert all(v is None for v in hashes(library.db).values())


def test_real_run_sets_hashes_reports_duplicates_and_failures(library, capsys):
    code = backfill_sha256.main([])
    out = capsys.readouterr().out
    got = hashes(library.db)
    assert code == 1  # d failed
    assert got["video a"] == h("same") and got["video b"] == h("other")
    assert got["video c"] is None  # same file as a: left empty, not deleted
    assert got["video d"] is None
    assert "DUPLICATE" in out and str(library.ids["a"]) in out and "FAILED" in out


def test_second_run_only_retries_what_is_still_empty(library):
    backfill_sha256.main([])
    library.sha.reset_mock()
    backfill_sha256.main([])
    hashed = {c[0][0] for c in library.sha.call_args_list}
    assert hashed == {"raw/c.mp4", "raw/d.mp4"}  # a and b already have hashes and are not read again


def test_prefers_the_untouched_original_over_a_remuxed_copy(library):
    """remux_faststart --keep-original keeps originals/<key>; its hash is the one a re-upload of the source would have."""
    library.objects["originals/raw/a.mp4"] = "original bytes"
    library.exists.side_effect = lambda key: key == "originals/raw/a.mp4"
    backfill_sha256.main(["--video-id", str(library.ids["a"])])
    assert hashes(library.db)["video a"] == h("original bytes")


def test_a_hash_that_already_belongs_to_a_video_is_never_overwritten(library):
    library.db.execute(text("UPDATE videos SET sha256 = :h WHERE title = 'video b'"), {"h": h("same")})
    library.db.commit()
    backfill_sha256.main([])
    got = hashes(library.db)
    assert got["video b"] == h("same") and got["video a"] is None and got["video c"] is None  # unique index respected


def test_nothing_to_do(api, db_session, capsys):
    assert backfill_sha256.main([]) == 0
    assert "Nothing to do" in capsys.readouterr().out
