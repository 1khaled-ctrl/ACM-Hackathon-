"""Optional S3 sync for model registry checkpoints.

Bucket name comes from the GOTRAIN_S3_BUCKET environment variable — not a
config key or argument — since where to store checkpoints is a deployment
concern, not something a run's hyperparameters should control. Both tools
fail gracefully (return a clear, structured result) rather than raising
when the bucket isn't configured or boto3 can't reach AWS, since this is
designed to work with no AWS access at all (the local, pre-SageMaker
case) as well as with it.
"""

import os
from pathlib import Path
from typing import Optional

from gotrain.registry import MODELS_DIR, get_version_dir

BUCKET_ENV_VAR = "GOTRAIN_S3_BUCKET"
S3_PREFIX = "gotrain-models"


def _bucket_name() -> Optional[str]:
    return os.environ.get(BUCKET_ENV_VAR) or None


def upload_to_s3(dataset: str, version: str) -> dict:
    """Upload a registered model version's files (models/<dataset>/<version>/)
    to s3://<bucket>/gotrain-models/<dataset>/<version>/.

    Returns {"status": "skipped", "reason": ...} if GOTRAIN_S3_BUCKET isn't
    set, or {"status": "error", "reason": ...} on an AWS/network failure —
    never raises, so a caller (agent or script) can check the result
    without a try/except.
    """
    bucket = _bucket_name()
    if bucket is None:
        return {
            "status": "skipped",
            "reason": f"{BUCKET_ENV_VAR} is not set — no S3 bucket configured.",
        }

    version_dir = get_version_dir(dataset, version)
    if not version_dir.is_dir():
        return {"status": "error", "reason": f"No local version directory at {version_dir}"}

    try:
        import boto3

        s3 = boto3.client("s3")
        uploaded = []
        for file_path in sorted(version_dir.iterdir()):
            if not file_path.is_file():
                continue
            key = f"{S3_PREFIX}/{dataset}/{version}/{file_path.name}"
            s3.upload_file(str(file_path), bucket, key)
            uploaded.append(key)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    return {"status": "success", "bucket": bucket, "uploaded_keys": uploaded}


def download_from_s3(dataset: str, version: str) -> dict:
    """Download a model version's files from
    s3://<bucket>/gotrain-models/<dataset>/<version>/ into
    models/<dataset>/<version>/ locally.

    Same graceful-failure contract as upload_to_s3.
    """
    bucket = _bucket_name()
    if bucket is None:
        return {
            "status": "skipped",
            "reason": f"{BUCKET_ENV_VAR} is not set — no S3 bucket configured.",
        }

    version_dir = get_version_dir(dataset, version)
    version_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{S3_PREFIX}/{dataset}/{version}/"

    try:
        import boto3

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        downloaded = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = Path(key).name
                if not filename:
                    continue
                dest = version_dir / filename
                s3.download_file(bucket, key, str(dest))
                downloaded.append(key)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    if not downloaded:
        return {
            "status": "error",
            "reason": f"No objects found at s3://{bucket}/{prefix}",
        }
    return {"status": "success", "bucket": bucket, "downloaded_keys": downloaded}
