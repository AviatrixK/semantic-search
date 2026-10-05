"""Production startup guards and the S3 client setup. Pure unit tests: no AWS, no services."""
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.core.config import Settings
from app.services import storage

GOOD = dict(APP_ENV="production", S3_ENDPOINT=None, S3_PUBLIC_ENDPOINT=None, S3_ACCESS_KEY=None, S3_SECRET_KEY=None, JWT_SECRET="x" * 48, COOKIE_SECURE=True,
            DATABASE_URL="postgresql+psycopg://svs:a1b2c3d4e5f6a7b8c9d0@postgres:5432/svs")


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch):
    """Settings reads os.environ (conftest, compose env_file, a developer's .env). Remove every setting so each test sees
    only the values it passes explicitly plus the code defaults."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)


def make(**over):
    return Settings(_env_file=None, **{**GOOD, **over})


def test_good_production_settings_start():
    s = make()
    assert s.is_production and s.S3_ENDPOINT is None


def test_default_region_is_ap_south_1():
    assert make().AWS_REGION == "ap-south-1"


@pytest.mark.parametrize("secret", ["change-me", "change-me-to-a-long-random-string", "", "short"])
def test_production_refuses_default_or_short_jwt_secret(secret):
    with pytest.raises(ValueError, match="JWT_SECRET"):
        make(JWT_SECRET=secret)


@pytest.mark.parametrize("pw", ["svs", "postgres", "password", ""])
def test_production_refuses_default_db_password(pw):
    with pytest.raises(ValueError, match="database password"):
        make(DATABASE_URL=f"postgresql+psycopg://svs:{pw}@postgres:5432/svs")


def test_production_requires_secure_cookie():
    with pytest.raises(ValueError, match="COOKIE_SECURE"):
        make(COOKIE_SECURE=False)


def test_development_keeps_old_defaults():
    s = Settings(_env_file=None, APP_ENV="development", JWT_SECRET="change-me", COOKIE_SECURE=False)
    assert not s.is_production


def test_blank_s3_values_mean_unset():
    s = make(S3_ENDPOINT="", S3_ACCESS_KEY=" ", S3_SECRET_KEY="")
    assert s.S3_ENDPOINT is None and s.S3_ACCESS_KEY is None


def _client_args(**over):
    with patch.object(storage, "settings", make(**over)), patch.object(storage.boto3, "client") as c:
        storage._client(storage.settings.S3_ENDPOINT)
    return c.call_args


def test_aws_client_uses_regional_endpoint_sigv4_and_default_credential_chain():
    call = _client_args(AWS_REGION="eu-west-2")
    assert call.kwargs["endpoint_url"] == "https://s3.eu-west-2.amazonaws.com"
    assert call.kwargs["region_name"] == "eu-west-2"
    assert call.kwargs["config"].signature_version == "s3v4"
    assert "aws_access_key_id" not in call.kwargs  # instance role via the default chain


def test_custom_endpoint_with_static_keys_for_local_dev():
    call = _client_args(S3_ENDPOINT="http://minio:9000", S3_ACCESS_KEY="a", S3_SECRET_KEY="b", AWS_REGION="us-east-1")
    assert call.kwargs["endpoint_url"] == "http://minio:9000"
    assert call.kwargs["aws_access_key_id"] == "a" and call.kwargs["aws_secret_access_key"] == "b"


def _head_error(code):
    return ClientError({"Error": {"Code": code}}, "HeadBucket")


def test_production_never_creates_a_missing_bucket():
    fake = MagicMock()
    fake.head_bucket.side_effect = _head_error("404")
    with patch.object(storage, "s3", fake), patch.object(storage, "settings", make()):
        with pytest.raises(storage.StorageError, match="does not exist"):
            storage.ensure_bucket()
    fake.create_bucket.assert_not_called()


def test_production_access_denied_is_a_clear_error():
    fake = MagicMock()
    fake.head_bucket.side_effect = _head_error("403")
    with patch.object(storage, "s3", fake), patch.object(storage, "settings", make()):
        with pytest.raises(storage.StorageError, match="instance role"):
            storage.ensure_bucket()


def test_development_still_creates_the_bucket():
    fake = MagicMock()
    fake.head_bucket.side_effect = _head_error("404")
    with patch.object(storage, "s3", fake), patch.object(storage, "settings", Settings(_env_file=None)):
        storage.ensure_bucket()
    fake.create_bucket.assert_called_once()
