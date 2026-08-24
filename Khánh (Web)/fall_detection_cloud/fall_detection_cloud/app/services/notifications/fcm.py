"""Gửi push notification qua Firebase Cloud Messaging (FCM).

Dùng Firebase Admin SDK. Nếu chưa cấu hình `FCM_CREDENTIALS_FILE` hoặc chưa
cài `firebase-admin`, chạy **dry-run** (chỉ log) để không chặn luồng.
"""

import logging
import os
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)

_initialized = False


@dataclass
class FCMResult:
    sent: int = 0
    failed: int = 0
    dry_run: bool = False
    error: str | None = None


def _ensure_app() -> bool:
    """Khởi tạo Firebase app (1 lần). Trả True nếu sẵn sàng gửi thật."""
    global _initialized
    if not (settings.FCM_ENABLED and settings.FCM_CREDENTIALS_FILE):
        return False
    if _initialized:
        return True
    if not os.path.exists(settings.FCM_CREDENTIALS_FILE):
        logger.warning(
            "FCM_CREDENTIALS_FILE không tồn tại: %s",
            settings.FCM_CREDENTIALS_FILE,
        )
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials

        cred = credentials.Certificate(settings.FCM_CREDENTIALS_FILE)
        firebase_admin.initialize_app(cred)
        _initialized = True
        return True
    except ImportError:
        logger.warning("Chưa cài firebase-admin; FCM chạy dry-run")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Khởi tạo Firebase thất bại: %s", exc)
        return False


def send_push(
    tokens: list[str], title: str, body: str, data: dict[str, str]
) -> FCMResult:
    """Gửi push tới danh sách device token. Không raise."""
    tokens = [t for t in tokens if t]
    if not tokens:
        logger.info("FCM: không có device token nào để gửi")
        return FCMResult(dry_run=True)

    if not _ensure_app():
        logger.info(
            "FCM dry-run: gửi tới %d token | title=%r", len(tokens), title
        )
        return FCMResult(dry_run=True)

    try:
        from firebase_admin import messaging

        message = messaging.MulticastMessage(
            tokens=tokens,
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in data.items()},
        )
        resp = messaging.send_each_for_multicast(message)
        return FCMResult(
            sent=resp.success_count, failed=resp.failure_count
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gửi FCM thất bại: %s", exc)
        return FCMResult(failed=len(tokens), error=str(exc))
