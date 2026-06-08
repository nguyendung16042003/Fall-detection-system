import uuid

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Event(Base):
    """Sự kiện ngã do Edge gửi lên."""

    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type = Column(String(50), default="fall_candidate", nullable=False)
    severity = Column(String(20), default="low", nullable=False)
    person_id = Column(Integer)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    bbox_json = Column(JSONB)
    state_before = Column(String(50))
    state_after = Column(String(50))
    transition_time_ms = Column(Integer)
    classification_confidence = Column(Float)
    fall_confidence = Column(Float)
    vlm_verdict = Column(String(50))
    vlm_confidence = Column(Float)
    image_url = Column(String(500))
    video_clip_url = Column(String(500))
    status = Column(String(20), default="new", nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    camera = relationship("Camera", back_populates="events")
    alerts = relationship(
        "Alert", back_populates="event", cascade="all, delete-orphan"
    )
