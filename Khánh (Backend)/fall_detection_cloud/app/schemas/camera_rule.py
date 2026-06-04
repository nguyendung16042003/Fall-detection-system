from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional

# Cấu trúc cơ bản cho các quy tắc cấu hình
class CameraRuleBase(BaseModel):
    # Ngưỡng tin cậy: giá trị từ 0.0 đến 1.0, mặc định là 0.5 [2]
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Ngưỡng độ tin cậy để xác định ngã")
    
    # Cửa sổ thời gian: mặc định 2 giây để phân biệt ngã thật và nằm nghỉ [3, 4]
    time_window_seconds: int = Field(2, ge=1, le=10, description="Thời gian (giây) quan sát sự thay đổi trạng thái")
    
    is_active: bool = True

# Schema dùng cho yêu cầu cập nhật (PUT /rules/{camera_id}) [1]
class CameraRuleUpdate(CameraRuleBase):
    pass

# Schema dùng để trả về dữ liệu (GET /rules/{camera_id}) [1]
class CameraRuleRead(CameraRuleBase):
    camera_id: UUID
    updated_at: datetime

    class Config:
        from_attributes = True # Cho phép Pydantic đọc dữ liệu trực tiếp từ SQLAlchemy Model