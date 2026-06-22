"""Lưu ảnh snapshot lên MinIO (S3-compatible).

Nếu chưa cấu hình `MINIO_ENDPOINT` hoặc chưa cài `minio`, mọi hàm trả None
(bỏ qua upload) để không chặn luồng xử lý sự kiện.
"""

import logging
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = None
_bucket_ready = False


def _is_configured() -> bool:
    return bool(
        settings.MINIO_ENDPOINT
        and settings.MINIO_ACCESS_KEY
        and settings.MINIO_SECRET_KEY
    )


def _get_client():
    global _client, _bucket_ready
    if not _is_configured():
        return None
    if _client is not None:
        return _client
    try:
        from minio import Minio
    except ImportError:
        logger.warning("Chưa cài 'minio'; bỏ qua upload snapshot")
        return None
    try:
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        if not _bucket_ready and not _client.bucket_exists(
            settings.MINIO_BUCKET
        ):
            _client.make_bucket(settings.MINIO_BUCKET)
        _bucket_ready = True
        return _client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không khởi tạo được MinIO client: %s", exc)
        _client = None
        return None


def upload_jpeg(data: bytes, key_prefix: str = "snapshots") -> str | None:
    """Upload JPEG, trả về URL công khai (hoặc None nếu không cấu hình)."""
    client = _get_client()
    if client is None or not data:
        return None
    import io

    name = f"{key_prefix}/{uuid.uuid4().hex}.jpg"
    try:
        client.put_object(
            settings.MINIO_BUCKET,
            name,
            io.BytesIO(data),
            length=len(data),
            content_type="image/jpeg",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Upload MinIO thất bại: %s", exc)
        return None

    base = settings.MINIO_PUBLIC_URL or (
        f"{'https' if settings.MINIO_SECURE else 'http'}://"
        f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}"
    )
    return f"{base.rstrip('/')}/{name}"


def upload_clip(frames: list[bytes], key_prefix: str = "clips") -> str | None:
    """Upload chuỗi frame JPEG như clip cho MLOps, trả về URL công khai."""
    client = _get_client()
    if client is None or not frames:
        return None
    
    # Tạo ZIP file chứa 6 frames
    import io
    import zipfile
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, frame in enumerate(frames):
            zip_file.writestr(f"frame_{i:03d}.jpg", frame)
    
    zip_data = zip_buffer.getvalue()
    name = f"{key_prefix}/{uuid.uuid4().hex}.zip"
    
    try:
        client.put_object(
            settings.MINIO_BUCKET,
            name,
            io.BytesIO(zip_data),
            length=len(zip_data),
            content_type="application/zip",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Upload clip MinIO thất bại: %s", exc)
        return None

    base = settings.MINIO_PUBLIC_URL or (
        f"{'https' if settings.MINIO_SECURE else 'http'}://"
        f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}"
    )
    return f"{base.rstrip('/')}/{name}"
