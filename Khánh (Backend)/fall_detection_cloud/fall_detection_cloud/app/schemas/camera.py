from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CameraCreate(BaseModel):
    """Tạo camera (contract v2 mục 2)."""

    cam_id: str = Field(..., examples=["cam_01"])
    name: str = Field(..., examples=["Phòng khách"])
    rtsp_url: str = Field(
        ..., examples=["rtsp://admin:pass@192.168.2.1:554/h264_stream"]
    )
    location: str | None = Field(default=None, examples=["Tầng 1"])
    edge_device_id: str | None = Field(default=None, examples=["jetson_nano_01"])
    is_active: bool = True


class CameraUpdate(BaseModel):
    name: str | None = None
    rtsp_url: str | None = None
    location: str | None = None
    edge_device_id: str | None = None
    is_active: bool | None = None


class CameraRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    cam_id: str
    name: str
    rtsp_url: str
    location: str | None = None
    is_active: bool
    edge_device_id: str | None = None
    status: str
    last_heartbeat: datetime | None = None
    created_at: datetime
