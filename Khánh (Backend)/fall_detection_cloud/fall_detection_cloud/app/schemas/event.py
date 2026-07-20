from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ClassBeforeEnum(str, Enum):
    """class_before enum theo mqtt_schema v3 - chỉ stand/sit."""
    STAND = "stand"
    SIT = "sit"


class FinalClassEnum(str, Enum):
    """final_class enum theo mqtt_schema v3 - tên gốc model SGIE."""
    STAND = "stand"
    SIT = "sit"
    LIE = "lie"
    BEND = "bend"
    EXERCISE = "exercise"
    HALF_PERSON = "half_person"


class TriggerEnum(str, Enum):
    """trigger enum theo mqtt_schema v3."""
    STAND_TO_LIE = "stand_to_lie"
    SIT_TO_LIE = "sit_to_lie"


class DetectionIn(BaseModel):
    """detection.* trong mqtt_schema v3 fall_event."""

    class_before: ClassBeforeEnum | None = None
    final_class: FinalClassEnum | None = None
    confidence: float | None = None
    bbox_xyxy: list[int] | None = Field(default=None, min_length=4, max_length=4)
    frame_width: int | None = None
    frame_height: int | None = None


class RuleIn(BaseModel):
    version: str | None = None
    trigger: TriggerEnum | None = None
    transition_ms: int | None = None
    window_ms: int | None = None


class FrameIn(BaseModel):
    offset_ms: int
    jpeg_b64: str


class EventCreate(BaseModel):
    """Payload Edge gửi (POST /events REST fallback) = mqtt_schema fall_event.

    Edge định danh camera bằng `cam_id` chuỗi (vd cam_01); server tra ra UUID
    camera. `event_id` (uuid Edge tạo) được dùng làm khóa chính nếu hợp lệ.
    
    Note: clip_url is optional for manual testing purposes (not in mqtt_schema).
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str | None = None
    event_id: str | None = None
    cam_id: str = Field(..., examples=["cam_01"])
    person_id: int | None = None
    timestamp_utc: datetime | None = Field(default=None, alias="timestamp")
    event_type: str = Field(default="fall_candidate")

    detection: DetectionIn | None = None
    rule: RuleIn | None = None
    frames: list[FrameIn] | None = None

    status: str = Field(default="pending")
    clip_url: str | None = None  # Optional for manual testing


class EventCreateResponse(BaseModel):
    id: int
    event_id: UUID


class DetectionOut(BaseModel):
    final_class: str | None = None
    confidence: float | None = None
    bbox_xyxy: list[int] | None = None


class VLMResultOut(BaseModel):
    fall: bool
    confidence: float | None = None
    reason: str | None = None


class EventOut(BaseModel):
    """Event object đầy đủ (contract v2 mục 3)."""

    id: int
    event_id: UUID
    cam_id: str | None = None
    person_id: int | None = None
    timestamp_utc: datetime
    event_type: str
    status: str
    detection: DetectionOut
    vlm_result: VLMResultOut | None = None
    clip_url: str | None = None
    snapshot_url: str | None = None
    created_at: datetime


class EventListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[EventOut]
