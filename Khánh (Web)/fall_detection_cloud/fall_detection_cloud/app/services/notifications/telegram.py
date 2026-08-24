"""Kênh thông báo phụ qua Telegram Bot.

Ưu tiên gửi kèm ảnh (sendPhoto) để người giám sát xem nhanh; nếu không có ảnh
thì gửi text (sendMessage). Chưa cấu hình token/chat_id sẽ chạy **dry-run**.
"""

import logging
from dataclasses import dataclass

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TelegramResult:
    ok: bool = False
    dry_run: bool = False
    error: str | None = None


def _is_configured() -> bool:
    return bool(
        settings.TELEGRAM_ENABLED
        and settings.TELEGRAM_BOT_TOKEN
        and settings.TELEGRAM_CHAT_ID
    )


def send_alert(
    caption: str,
    photo_url: str | None = None,
    photo_bytes: bytes | None = None,
) -> TelegramResult:
    """Gửi cảnh báo (kèm ảnh nếu có). Không raise."""
    if not _is_configured():
        logger.info("Telegram dry-run | caption=%r photo=%s", caption, photo_url)
        return TelegramResult(dry_run=True)

    base = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}"
    chat_id = settings.TELEGRAM_CHAT_ID

    try:
        if photo_bytes:
            resp = httpx.post(
                f"{base}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": ("snapshot.jpg", photo_bytes, "image/jpeg")},
                timeout=10.0,
            )
        elif photo_url and photo_url.startswith(("http://", "https://")):
            resp = httpx.post(
                f"{base}/sendPhoto",
                data={
                    "chat_id": chat_id,
                    "photo": photo_url,
                    "caption": caption,
                },
                timeout=10.0,
            )
        else:
            resp = httpx.post(
                f"{base}/sendMessage",
                data={"chat_id": chat_id, "text": caption},
                timeout=10.0,
            )
        resp.raise_for_status()
        return TelegramResult(ok=True)
    except httpx.HTTPError as exc:
        logger.warning("Gửi Telegram thất bại: %s", exc)
        return TelegramResult(error=str(exc))
