from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class EventCreate(BaseModel):
    """Payload Edge (Jetson) gửi lên qua POST /events.

    Bám theo api_contract.md mục 3.1. Một số field chỉ phục vụ Edge
    (schema_version, edge_device_id, source_id, rule_version) được chấp nhận
    nhưng không lưu DB vì bảng `events` (db.txt) không có cột tương ứng.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Định danh / metadata phía Edge (tùy chọn, không bắt buộc lưu)
    schema_version: str | None = None
    event_id: str | None = Field(
        default=None, examples=["evt_20260531_000001"]
    )
    edge_device_id: str | None = Field(default=None, examples=["jetson_nano_01"])
    source_id: int | None = None
    rule_version: str | None = None

    # Dữ liệu sự kiện (lưu vào bảng events)
    camera_id: UUID = Field(..., description="UUID camera trong DB")
    event_type: str = Field(default="fall_candidate")
    # Bí danh theo mô tả task: chấp nhận class_label thay cho event_type
    class_label: str | None = Field(default=None, exclude=True)
    severity: str = Field(default="low", examples=["high", "low"])
    person_id: int | None = None
    timestamp: datetime
    bbox: BBox | None = None
    state_before: str | None = None
    state_after: str | None = None
    transition_time_ms: int | None = None
    classification_confidence: float | None = None
    fall_confidence: float | None = None
    # Bí danh theo mô tả task: confidence -> fall_confidence
    confidence: float | None = Field(default=None, exclude=True)
    vlm_verdict: str | None = None
    vlm_confidence: float | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    status: str = Field(default="new")


class EventCreateResponse(BaseModel):
    event_id: str
    status: str = "received"


class EventListItem(BaseModel):
    event_id: UUID
    camera_id: UUID
    camera_name: str | None = None
    timestamp: datetime
    severity: str
    status: str
    fall_confidence: float | None = None
    snapshot_url: str | None = None


class EventListResponse(BaseModel):
    items: list[EventListItem]
    page: int
    limit: int
    total: int


class EventDetail(BaseModel):
    event_id: UUID
    event_type: str
    severity: str
    camera_id: UUID
    camera_name: str | None = None
    person_id: int | None = None
    timestamp: datetime
    bbox: BBox | None = None
    state_before: str | None = None
    state_after: str | None = None
    transition_time_ms: int | None = None
    classification_confidence: float | None = None
    fall_confidence: float | None = None
    vlm_verdict: str | None = None
    vlm_confidence: float | None = None
    snapshot_url: str | None = None
    clip_url: str | None = None
    status: str
