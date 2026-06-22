"""MinIO Bucket Setup Script.

Tạo các bucket cần thiết và cấu hình policy cho presigned URL:
- fall-events: lưu snapshots và clips
- false-positives: lưu clips cho MLOps
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bucket_policy(bucket_name: str) -> str:
    """Tạo policy cho bucket cho phép presigned URL."""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            }
        ],
    }
    return json.dumps(policy)


def setup_minio():
    """Setup MinIO buckets và policies."""
    # Lấy cấu hình từ environment hoặc dùng mặc định
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ROOT_USER", "admin")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD", "password123")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    logger.info(f"Kết nối đến MinIO tại {endpoint}")
    
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )

    buckets = [
        "fall-events",
        "false-positives",
        "clips",
    ]

    for bucket in buckets:
        try:
            # Kiểm tra bucket đã tồn tại chưa
            if not client.bucket_exists(bucket):
                logger.info(f"Tạo bucket: {bucket}")
                client.make_bucket(bucket)
                
                # Tạo policy cho presigned URL
                policy = create_bucket_policy(bucket)
                client.set_bucket_policy(bucket, policy)
                logger.info(f"Đã cấu hình policy cho bucket: {bucket}")
            else:
                logger.info(f"Bucket {bucket} đã tồn tại, cập nhật policy...")
                policy = create_bucket_policy(bucket)
                client.set_bucket_policy(bucket, policy)
                logger.info(f"Đã cập nhật policy cho bucket: {bucket}")
                
        except S3Error as exc:
            logger.error(f"Lỗi khi setup bucket {bucket}: {exc}")
            return False

    logger.info("MinIO setup hoàn tất!")
    
    # Hiển thị thông tin kết nối
    logger.info("=" * 60)
    logger.info("MinIO Connection Info:")
    logger.info(f"  Endpoint: {endpoint}")
    logger.info(f"  Access Key: {access_key}")
    logger.info(f"  Buckets: {', '.join(buckets)}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = setup_minio()
    sys.exit(0 if success else 1)
