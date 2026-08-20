"""Tiếp nhận & lưu sự kiện ngã từ Edge (dùng chung cho REST và MQTT).

Map mqtt_schema v2 `fall_event` -> bảng `events`, tra cứu camera theo `cam_id`,
giải mã frame trigger (offset 0) và upload snapshot lên MinIO nếu có cấu hình.
"""

import base64
import binascii
import logging
import uuid

from sqlalchemy.orm import Session

from app.core.storage import upload_jpeg
from app.models.camera import Camera
from app.models.event import Event
from app.schemas.event import EventCreate

logger = logging.getLogger(__name__)


class CameraNotFoundError(Exception):
    """Không tìm thấy camera theo cam_id."""


def _decode_trigger_frame(data: EventCreate) -> bytes | None:
    """Lấy frame tại offset_ms=0 (lúc phát hiện ngã) và giải mã base64."""
    if not data.frames:
        return None
    frame = next(
        (f for f in data.frames if f.offset_ms == 0), data.frames[-1]
    )
    try:
        return base64.b64decode(frame.jpeg_b64)
    except (binascii.Error, ValueError) as exc:
        logger.warning("Frame base64 không hợp lệ: %s", exc)
        return None


def _decode_all_frames_with_offsets(
    data: EventCreate,
) -> list[tuple[int, bytes]] | None:
    """Giải mã 6 frame kèm offset_ms, sắp theo offset_ms tăng dần."""
    if not data.frames or len(data.frames) != 6:
        logger.warning(
            "Cần đúng 6 frame để lưu ảnh bằng chứng, nhận được %d",
            len(data.frames) if data.frames else 0,
        )
        return None

    result: list[tuple[int, bytes]] = []
    for frame in sorted(data.frames, key=lambda f: f.offset_ms):
        try:
            result.append((frame.offset_ms, base64.b64decode(frame.jpeg_b64)))
        except (binascii.Error, ValueError) as exc:
            logger.warning(
                "Frame base64 không hợp lệ tại offset_ms=%d: %s",
                frame.offset_ms,
                exc,
            )
            return None

    if len(result) != 6:
        logger.warning("Giải mã được %d/6 frame", len(result))
        return None

    return result


def _decode_all_frames(data: EventCreate) -> list[bytes] | None:
    """Giải mã tất cả 6 frame từ payload cho VLM temporal analysis."""
    pairs = _decode_all_frames_with_offsets(data)
    return [frame_bytes for _, frame_bytes in pairs] if pairs else None


def ingest_event(
    db: Session, data: EventCreate
) -> tuple[Event, bytes | None, list[bytes] | None]:
    """Lưu một sự kiện ngã, trả về (Event, bytes frame trigger, list 6 frames)."""
    camera = db.query(Camera).filter(Camera.cam_id == data.cam_id).first()
    if camera is None:
        raise CameraNotFoundError(data.cam_id)

    # event_id do Edge tạo (uuid v4) -> lưu vào cột event_id; id là PK int
    event_uuid: uuid.UUID | None = None
    if data.event_id:
        try:
            event_uuid = uuid.UUID(str(data.event_id))
        except ValueError:
            event_uuid = None

    det = data.detection
    bbox_json = None
    if det and det.bbox_xyxy and len(det.bbox_xyxy) == 4:
        x1, y1, x2, y2 = det.bbox_xyxy
        bbox_json = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    frame_bytes = _decode_trigger_frame(data)
    all_frames_with_offsets = _decode_all_frames_with_offsets(data)
    all_frames = (
        [b for _, b in all_frames_with_offsets] if all_frames_with_offsets else None
    )
    image_url = upload_jpeg(frame_bytes) if frame_bytes else None

    image_urls_json = None
    if all_frames_with_offsets:
        entries = []
        for idx, (offset_ms, frame_data) in enumerate(all_frames_with_offsets):
            # offset_ms==0 là frame trigger đã upload ở trên -> tái dùng URL,
            # tránh upload trùng cùng 1 ảnh lên MinIO 2 lần.
            url = (
                image_url
                if (offset_ms == 0 and image_url)
                else upload_jpeg(frame_data)
            )
            entries.append({"index": idx, "offset_ms": offset_ms, "url": url})
        image_urls_json = entries

    event = Event(
        event_id=event_uuid or uuid.uuid4(),
        camera_id=camera.id,
        event_type=data.event_type or "fall_candidate",
        person_id=data.person_id,
        timestamp_utc=data.timestamp_utc,
        class_before=det.class_before if det else None,
        final_class=det.final_class if det else None,
        detection_confidence=det.confidence if det else None,
        bbox_json=bbox_json,
        frame_width=det.frame_width if det else None,
        frame_height=det.frame_height if det else None,
        rule_trigger=data.rule.trigger if data.rule else None,
        transition_ms=data.rule.transition_ms if data.rule else None,
        image_url=image_url,
        image_urls=image_urls_json,
        clip_url=data.clip_url,  # For manual testing purposes
        status=data.status or "pending",
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event, frame_bytes, all_frames
