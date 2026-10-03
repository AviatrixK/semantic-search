import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services import retrieval, storage


def test_put_json_writes_utf8_object():
    with patch.object(storage, "s3") as s3:
        storage.put_json("transcripts/abc.json", {"language": "en", "segments": [{"text": "café"}]})
    kw = s3.put_object.call_args.kwargs
    assert kw["Key"] == "transcripts/abc.json" and kw["ContentType"] == "application/json"
    assert json.loads(kw["Body"].decode("utf-8"))["segments"][0]["text"] == "café"


def test_list_chunks_maps_rows():
    db = MagicMock()
    db.scalars.return_value.all.return_value = [SimpleNamespace(idx=0, start_sec=0.0, end_sec=30.0, text="a"),
                                                SimpleNamespace(idx=1, start_sec=25.0, end_sec=55.0, text="b")]
    out = retrieval.list_chunks(db, uuid.uuid4())
    assert out == [{"idx": 0, "start_sec": 0.0, "end_sec": 30.0, "text": "a"},
                   {"idx": 1, "start_sec": 25.0, "end_sec": 55.0, "text": "b"}]
