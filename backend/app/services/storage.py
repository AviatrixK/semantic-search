"""S3 storage: AWS S3 in production (S3_ENDPOINT unset), SeaweedFS locally (S3_ENDPOINT set)."""
import json

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings


def _client(endpoint: str | None):
    """endpoint=None -> the regional AWS endpoint (https://s3.<region>.amazonaws.com), so neither normal calls nor
    pre-signed URLs ever hit a cross-region redirect. SigV4 everywhere. Credentials: explicit keys only when both are set
    (local SeaweedFS); otherwise boto3's default chain (the EC2 instance role)."""
    kwargs = {}
    if settings.S3_ACCESS_KEY and settings.S3_SECRET_KEY:
        kwargs.update(aws_access_key_id=settings.S3_ACCESS_KEY, aws_secret_access_key=settings.S3_SECRET_KEY)
    if endpoint:
        config = Config(signature_version="s3v4")
    else:
        endpoint = f"https://s3.{settings.AWS_REGION}.amazonaws.com"
        config = Config(signature_version="s3v4", s3={"addressing_style": "virtual"})
    return boto3.client("s3", endpoint_url=endpoint, config=config, region_name=settings.AWS_REGION, **kwargs)


s3 = _client(settings.S3_ENDPOINT)
# Signs URLs the browser can reach. Production: same client (the bucket is private; the URL is pre-signed).
s3_public = _client(settings.S3_PUBLIC_ENDPOINT) if settings.S3_PUBLIC_ENDPOINT else s3


class StorageError(RuntimeError):
    """Startup storage problem with a message an operator can act on."""


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=settings.S3_BUCKET)
        return
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if not settings.is_production:
            s3.create_bucket(Bucket=settings.S3_BUCKET)
            return
        if code in ("404", "NoSuchBucket", "NotFound"):
            raise StorageError(f"S3 bucket '{settings.S3_BUCKET}' does not exist in {settings.AWS_REGION}. "
                               "Create it (private) before starting; production never creates buckets.") from exc
        raise StorageError(f"Cannot access S3 bucket '{settings.S3_BUCKET}' (error {code or 'unknown'}). Check that the "
                           "EC2 instance role allows s3:ListBucket on the bucket and that the name/region are right.") from exc
    except Exception as exc:  # no credentials, endpoint unreachable...
        if not settings.is_production:
            raise
        raise StorageError(f"Cannot reach S3 bucket '{settings.S3_BUCKET}': {type(exc).__name__}: {exc}. "
                           "'Unable to locate credentials' means the instance role is not reachable from the container "
                           "(attach the role; set IMDS hop limit to 2).") from exc


def upload_fileobj(fileobj, key: str, content_type: str):
    s3.upload_fileobj(fileobj, settings.S3_BUCKET, key, ExtraArgs={"ContentType": content_type})


def put_json(key: str, obj) -> None:
    s3.put_object(Bucket=settings.S3_BUCKET, Key=key, ContentType="application/json",
                  Body=json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def read_head(key: str, nbytes: int = 65536) -> bytes:
    """The first `nbytes` of an object (a ranged GET): enough to read an MP4's top-level box headers without downloading it."""
    obj = s3.get_object(Bucket=settings.S3_BUCKET, Key=key, Range=f"bytes=0-{nbytes - 1}")
    return obj["Body"].read()


def object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=settings.S3_BUCKET, Key=key)
        return True
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code", "")) in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def sha256_of_object(key: str) -> str:
    """Hex SHA-256 of a stored object, streamed in 1 MiB pieces (never held in memory)."""
    import hashlib
    digest = hashlib.sha256()
    for block in s3.get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"].iter_chunks(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def download_to(key: str, path: str):
    s3.download_file(settings.S3_BUCKET, key, path)


def presigned_url(key: str, expires: int = 3600) -> str:
    return s3_public.generate_presigned_url("get_object", Params={"Bucket": settings.S3_BUCKET, "Key": key},
                                            ExpiresIn=expires)


def delete(key: str):
    s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)
