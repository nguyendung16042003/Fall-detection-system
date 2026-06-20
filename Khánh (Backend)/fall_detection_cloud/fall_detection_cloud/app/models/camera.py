from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Contract v2: Edge định danh camera bằng chuỗi cam_id (vd "cam_01")
    cam_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String(255), nullable=False)
    location = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    # Trường nội bộ (không thuộc contract object nhưng phục vụ vận hành)
    edge_device_id = Column(String(100), nullable=False, default="")
    status = Column(String(20), default="unknown", nullable=False)
    last_heartbeat = Column(DateTime(timezone=True))
    note = Column(Text)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    rules = relationship(
        "CameraRule", back_populates="camera", cascade="all, delete-orphan"
    )
    events = relationship(
        "Event", back_populates="camera", cascade="all, delete-orphan"
    )
