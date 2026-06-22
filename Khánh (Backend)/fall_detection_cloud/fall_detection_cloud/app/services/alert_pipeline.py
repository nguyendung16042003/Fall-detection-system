"""Trigger chain: Event -> VLM verify -> tạo Alert -> gửi FCM + Telegram.

Luồng (contract v2 — mọi event đều qua VLM, không còn confidence routing):
1. Lấy `camera_rules` của camera; nếu `enable_vlm_verify=TRUE` thì gọi VLM
   xác minh ảnh, ghi `events.vlm_verdict` / `vlm_confidence` / `vlm_reason`.
2. Nếu VLM kết luận `not_fall` -> `event.status=false_positive`, bỏ qua
   (giảm báo động giả), không tạo alert.
3. Ngược lại: `event.status=confirmed`, tạo bản ghi `alerts`, gửi push FCM tới
   token đã đăng ký và gửi Telegram (kèm ảnh). Mọi kênh fail-safe (dry-run khi
   thiếu cấu hình).
4. Dedup multi-camera: gộp sự kiện từ 2 camera cách nhau < 2s thành 1 sự kiện.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.models.alert import Alert
from app.models.camera_rule import CameraRule
from app.models.event import Event
from app.models.user_device import UserDevice
from app.services import vlm
from app.services.notifications import fcm, telegram
from app.services.dedup import get_deduplicator, find_nearby_events, merge_events_metadata
from app.core.storage import upload_clip

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    event_id: str
    vlm_ran: bool = False
    vlm_verdict: str | None = None
    vlm_confidence: float | None = None
    vlm_latency_ms: float | None = None
    alert_created: bool = False
    alert_id: str | None = None
    suppressed: bool = False
    fcm_sent: int = 0
    telegram_ok: bool = False
    notes: list[str] = field(default_factory=list)


def _build_caption(event: Event, camera_name: str | None) -> str:
    conf = (
        event.detection_confidence
        if event.detection_confidence is not None
        else 0.0
    )
    lines = [
        "🚨 CẢNH BÁO NGÃ",
        f"Camera: {camera_name or event.camera_id}",
        f"Thời gian: {event.timestamp_utc:%Y-%m-%d %H:%M:%S}",
        f"Độ tin cậy: {conf:.0%}",
    ]
    if event.vlm_verdict:
        lines.append(
            f"VLM: {event.vlm_verdict} ({(event.vlm_confidence or 0):.0%})"
        )
    if event.vlm_reason:
        lines.append(f"Lý do: {event.vlm_reason}")
    if event.image_url:
        lines.append(f"Ảnh: {event.image_url}")
    return "\n".join(lines)


def _active_fcm_tokens(db: Session) -> list[str]:
    rows = (
        db.query(UserDevice.fcm_token)
        .filter(UserDevice.fcm_token.isnot(None))
        .all()
    )
    return [r[0] for r in rows if r[0]]


def run_pipeline(
    db: Session, event: Event, image_bytes: bytes | None = None, frames_bytes: list[bytes] | None = None
) -> PipelineResult:
    result = PipelineResult(event_id=str(event.id))
    camera_name = event.camera.name if event.camera else None

    # 0. Dedup multi-camera events (trước khi xử lý VLM)
    deduplicator = get_deduplicator()
    merged_group = deduplicator.add_event(event)
    
    if merged_group and len(merged_group) > 1:
        # Event đã được gộp với event khác
        result.notes.append(f"Event gộp với {len(merged_group) - 1} event khác trong window 2s")
        # Chỉ xử lý VLM cho representative event (event đầu tiên trong group)
        if event.id != merged_group[0]:
            # Đây là event phụ, đánh dấu merged và skip
            event.status = "merged"
            db.add(event)
            db.commit()
            result.suppressed = True
            result.notes.append("Event phụ đã gộp, skip VLM và alert")
            return result
    
    rule = (
        db.query(CameraRule)
        .filter(CameraRule.camera_id == event.camera_id)
        .first()
    )

    # 1. VLM verify (nếu rule cho phép)
    if rule is not None and rule.enable_vlm_verify:
        vlm_res = vlm.verify_fall(
            image_bytes=image_bytes, 
            image_ref=event.image_url,
            frames_bytes=frames_bytes,
        )
        result.vlm_ran = True
        result.vlm_verdict = vlm_res.verdict
        result.vlm_confidence = vlm_res.confidence
        result.vlm_latency_ms = vlm_res.latency_ms
        event.vlm_verdict = vlm_res.verdict
        event.vlm_confidence = vlm_res.confidence
        event.vlm_reason = vlm_res.reason or None
        db.add(event)
        db.commit()
        if vlm_res.dry_run:
            result.notes.append("VLM dry-run")
        if vlm_res.error:
            result.notes.append(f"VLM error: {vlm_res.error}")

        # 2. Giảm báo động giả: VLM khẳng định không ngã -> bỏ qua
        if vlm_res.verdict == vlm.VERDICT_NOT_FALL:
            event.status = "false_positive"
            
            # MLOps: Lưu clip cho false positives để phân tích và tái huấn luyện
            if frames_bytes and len(frames_bytes) == 6:
                clip_url = upload_clip(frames_bytes, key_prefix="false_positives")
                if clip_url:
                    event.clip_url = clip_url
                    result.notes.append(f"MLOps: clip lưu tại {clip_url}")
            
            db.add(event)
            db.commit()
            result.suppressed = True
            result.notes.append("VLM: not_fall -> bỏ qua, không gửi cảnh báo")
            return result
    else:
        result.notes.append("Bỏ qua VLM (enable_vlm_verify=false hoặc no rule)")

    # 3. Xác nhận ngã -> tạo alert
    event.status = "confirmed"
    db.add(event)
    db.commit()

    title = f"Phát hiện ngã tại {camera_name or 'camera'}"
    alert = Alert(
        event_id=event.event_id,
        title=title,
        message=_build_caption(event, camera_name),
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    result.alert_created = True
    result.alert_id = str(alert.id)

    # 4. Gửi thông báo (FCM + Telegram)
    data = {
        "event_id": str(event.event_id),
        "alert_id": str(alert.id),
        "cam_id": event.camera.cam_id if event.camera else "",
        "type": "fall_alert",
    }
    fcm_res = fcm.send_push(
        tokens=_active_fcm_tokens(db),
        title=title,
        body=alert.message,
        data=data,
    )
    result.fcm_sent = fcm_res.sent
    alert.fcm_sent = bool(fcm_res.sent) or fcm_res.dry_run
    db.add(alert)
    db.commit()
    if fcm_res.dry_run:
        result.notes.append("FCM dry-run")

    tg_res = telegram.send_alert(
        caption=alert.message, photo_url=event.image_url, photo_bytes=image_bytes
    )
    result.telegram_ok = tg_res.ok
    if tg_res.dry_run:
        result.notes.append("Telegram dry-run")

    return result
