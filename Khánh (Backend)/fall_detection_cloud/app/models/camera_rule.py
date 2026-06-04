from sqlalchemy import Column, Integer, Float, ForeignKey, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base

class CameraRule(Base):
    __tablename__ = "camera_rules"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"))
    
    # Ngưỡng độ tin cậy để kích hoạt cảnh báo (mặc định 0.5) [1]
    confidence_threshold = Column(Float, default=0.5)
    
    # Cửa sổ thời gian để phân biệt ngã vs nằm nghỉ (mặc định 2 giây) [2, 5]
    time_window_seconds = Column(Integer, default=2)
    
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())