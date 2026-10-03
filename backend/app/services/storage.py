"""S3-compatible storage (SeaweedFS locally, S3/R2 in the demo)."""
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings


def _client(endpoint: str):
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=settings.S3_ACCESS_KEY,
                        aws_secret_access_key=settings.S3_SECRET_KEY,
                        config=Config(signature_version="s3v4"), region_name="us-east-1")


s3 = _client(settings.S3_ENDPOINT)
s3_public = _client(settings.S3_PUBLIC_ENDPOINT)  # signs URLs the browser can reach


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=settings.S3_BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=settings.S3_BUCKET)


def upload_fileobj(fileobj, key: str, content_type: str):
    s3.upload_fileobj(fileobj, settings.S3_BUCKET, key, ExtraArgs={"ContentType": content_type})


def download_to(key: str, path: str):
    s3.download_file(settings.S3_BUCKET, key, path)


def presigned_url(key: str, expires: int = 3600) -> str:
    return s3_public.generate_presigned_url("get_object", Params={"Bucket": settings.S3_BUCKET, "Key": key},
                                            ExpiresIn=expires)


def delete(key: str):
    s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)
