from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional

# Cấu trúc cơ bản chung cho Camera
class CameraBase(BaseModel):
    name: str = Field(..., example="Camera Phòng Khách")
    rtsp_url: str = Field(..., example="rtsp://admin:password@192.168.1.100:554/stream")
    location_note: Optional[str] = None

# Dữ liệu nhận vào khi tạo mới (POST /cameras)
class CameraCreate(CameraBase):
    pass

# Dữ liệu nhận vào khi cập nhật (PUT /cameras/{id})
class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location_note: Optional[str] = None
    status: Optional[str] = None

# Dữ liệu trả về cho Client (GET /cameras)
class CameraRead(CameraBase):
    id: UUID
    status: str
    last_heartbeat: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True