from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CameraBase(BaseModel):
    name: str = Field(..., examples=["Camera Phòng Khách"])
    rtsp_url: str = Field(
        ..., examples=["rtsp://admin:pass@192.168.1.100:554/stream"]
    )
    edge_device_id: str = Field(..., examples=["jetson_nano_01"])
    location_note: str | None = None


class CameraCreate(CameraBase):
    status: str | None = Field(default="unknown")


class CameraUpdate(BaseModel):
    name: str | None = None
    rtsp_url: str | None = None
    edge_device_id: str | None = None
    location_note: str | None = None
    status: str | None = None
    # Bí danh theo API contract: enabled=true -> "online", false -> "offline"
    enabled: bool | None = None


class CameraRead(CameraBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    last_heartbeat: datetime | None = None
    created_at: datetime
