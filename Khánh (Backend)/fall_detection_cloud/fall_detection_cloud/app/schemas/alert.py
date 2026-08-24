from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AlertOut(BaseModel):
    """Alert object (contract v2 mục 4)."""

    id: int
    event_id: UUID
    cam_id: str | None = None
    timestamp_utc: datetime | None = None
    acknowledged: bool
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    fcm_sent: bool = False
    vlm_reason: str | None = None


class AlertListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AlertOut]


class AlertAckResponse(BaseModel):
    id: int
    acknowledged: bool = True
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
