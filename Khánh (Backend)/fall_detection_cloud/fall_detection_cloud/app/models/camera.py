import uuid

from sqlalchemy import Column, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String(255), nullable=False)
    edge_device_id = Column(String(100), nullable=False)
    location_note = Column(Text)
    status = Column(String(20), default="unknown", nullable=False)
    last_heartbeat = Column(DateTime(timezone=True))
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    rules = relationship(
        "CameraRule", back_populates="camera", cascade="all, delete-orphan"
    )
    events = relationship(
        "Event", back_populates="camera", cascade="all, delete-orphan"
    )
