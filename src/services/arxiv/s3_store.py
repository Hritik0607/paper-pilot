import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_pdf(local_path: Path, bucket: str, key: str, region: str) -> bool:
    try:
        import boto3

        boto3.client("s3", region_name=region).upload_file(str(local_path), bucket, key)
        return True
    except Exception as e:
        logger.warning(f"S3 upload failed for {key}: {e}")
        return False


def fetch_pdf(local_path: Path, bucket: str, key: str, region: str) -> bool:
    try:
        import boto3

        boto3.client("s3", region_name=region).download_file(bucket, key, str(local_path))
        return True
    except Exception as e:
        logger.info(f"Not taken from S3 ({key}): {e}")
        return False
    