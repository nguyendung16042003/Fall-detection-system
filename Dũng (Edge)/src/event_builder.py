#!/usr/bin/env python3
"""Dựng payload JSON đúng mqtt_schema v3 (nội bộ schema_version 1.2).

Tách riêng khỏi pipeline/mqtt để test được không cần GPU/broker.
Chuẩn tham chiếu: mqtt_schema_v3.json ở gốc repo.

Hàm chính:
  - build_fall_event(candidate, frames)  -> dict  (topic events/cam_{id}/fall)
  - build_telemetry(cam_id, pipeline=, system=, network=) -> dict
  - fall_topic(cam_id) / telemetry_topic(cam_id)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, Union

from fall_detector import FallCandidate

SCHEMA_VERSION = "1.2"
EVENT_TYPE = "fall_candidate"
RULE_VERSION = "1.0"

# Enum đồng bộ với _validation_rules trong mqtt_schema_v3.json
# "unknown"/"sustained_lying" MỚI thêm theo fall_rule.py bản Tấn Dũng gửi
# (Handoff_for_Edge 2026-08-08) — CHƯA xác nhận với Khánh, xem CLAUDE.md "SGIE v2".
CLASS_BEFORE_ENUM = ("stand", "sit", "unknown")
FINAL_CLASS_ENUM = ("stand", "sit", "lie", "bend", "exercise", "half_person")
TRIGGER_ENUM = ("stand_to_lie", "sit_to_lie", "sustained_lying")
REQUIRED_OFFSETS = (-2500, -2000, -1500, -1000, -500, 0)
CAM_ID_RE = re.compile(r"^cam_[0-9]+$")

FrameLike = Union[Dict[str, object], Tuple[int, str]]

# 1x1 JPEG hợp lệ (đỏ) — CHỈ dùng tạm khi CHƯA có rolling buffer (Tuần 5).
# Ai gọi placeholder_frames() PHẢI tự log cảnh báo là ảnh GIẢ — không được lặng lẽ gửi.
PLACEHOLDER_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
    "AAAAAAAAAAAAAAAAAAAAA//EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AfwD/2Q=="
)


def placeholder_frames() -> List[Dict[str, object]]:
    """6 frame GIẢ đúng offset schema — dùng tạm tới khi có rolling buffer (Tuần 5).
    KHÔNG dùng cho production; nơi gọi phải log cảnh báo rõ ràng."""
    return [{"offset_ms": o, "jpeg_b64": PLACEHOLDER_JPEG_B64} for o in REQUIRED_OFFSETS]


def to_iso_z(dt: datetime) -> str:
    """ISO 8601 UTC với hậu tố 'Z' và mili-giây, vd 2026-06-15T10:30:00.123Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def now_iso_z() -> str:
    return to_iso_z(datetime.now(timezone.utc))


def fall_topic(cam_id: str) -> str:
    return f"events/{cam_id}/fall"


def telemetry_topic(cam_id: str) -> str:
    return f"telemetry/{cam_id}/status"


def _normalize_frame(f: FrameLike) -> Dict[str, object]:
    if isinstance(f, dict):
        return {"offset_ms": int(f["offset_ms"]), "jpeg_b64": str(f["jpeg_b64"])}
    offset_ms, jpeg_b64 = f  # tuple (offset_ms, jpeg_b64)
    return {"offset_ms": int(offset_ms), "jpeg_b64": str(jpeg_b64)}


def build_fall_event(
    candidate: FallCandidate,
    frames: Sequence[FrameLike],
    *,
    event_id: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
) -> Dict[str, object]:
    """Dựng payload fall_event. Kiểm tra đầu vào ngặt (fail nhanh) để không đẩy
    payload sai schema sang Khánh."""
    # --- validate candidate ---
    if not CAM_ID_RE.match(candidate.cam_id):
        raise ValueError(f"cam_id sai pattern cam_[0-9]+: {candidate.cam_id!r}")
    if candidate.class_before not in CLASS_BEFORE_ENUM:
        raise ValueError(f"class_before ngoài enum: {candidate.class_before!r}")
    if candidate.final_class not in FINAL_CLASS_ENUM:
        raise ValueError(f"final_class ngoài enum: {candidate.final_class!r}")
    if candidate.trigger not in TRIGGER_ENUM:
        raise ValueError(f"trigger ngoài enum: {candidate.trigger!r}")
    if not (0.0 <= candidate.confidence <= 1.0):
        raise ValueError(f"confidence phải trong [0,1]: {candidate.confidence}")
    if int(candidate.person_id) < 0:
        raise ValueError(f"person_id phải >= 0: {candidate.person_id}")

    # --- validate frames ---
    norm: List[Dict[str, object]] = [_normalize_frame(f) for f in frames]
    offsets = tuple(f["offset_ms"] for f in norm)
    if offsets != REQUIRED_OFFSETS:
        raise ValueError(
            f"frames phải đúng 6 phần tử offset {REQUIRED_OFFSETS}, nhận {offsets}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or str(uuid.uuid4()),
        "cam_id": candidate.cam_id,
        "person_id": int(candidate.person_id),
        "timestamp_utc": timestamp_utc or now_iso_z(),
        "event_type": EVENT_TYPE,
        "detection": {
            "class_before": candidate.class_before,
            "final_class": candidate.final_class,
            "confidence": round(float(candidate.confidence), 4),
            "bbox_xyxy": list(candidate.bbox_xyxy),
            "frame_width": int(candidate.frame_width),
            "frame_height": int(candidate.frame_height),
        },
        "rule": {
            "version": RULE_VERSION,
            "trigger": candidate.trigger,
            "transition_ms": int(candidate.transition_ms),
            "window_ms": int(candidate.window_ms),
        },
        "frames": norm,
    }


def build_telemetry(
    cam_id: str,
    *,
    pipeline: Dict[str, object],
    system: Dict[str, object],
    network: Dict[str, object],
    timestamp_utc: Optional[str] = None,
) -> Dict[str, object]:
    """Dựng payload telemetry (gửi mỗi 30s lên telemetry/cam_{id}/status)."""
    if not CAM_ID_RE.match(cam_id):
        raise ValueError(f"cam_id sai pattern cam_[0-9]+: {cam_id!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "cam_id": cam_id,
        "timestamp_utc": timestamp_utc or now_iso_z(),
        "pipeline": dict(pipeline),
        "system": dict(system),
        "network": dict(network),
    }
