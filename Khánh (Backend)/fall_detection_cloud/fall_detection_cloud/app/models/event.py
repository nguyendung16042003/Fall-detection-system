import uuid

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Event(Base):
    """Sự kiện ngã do Edge gửi lên (contract v2 mục 3)."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # event_id: UUID v4 do Edge tạo (CHANGE-2) — unique, dùng đối soát/MQTT
    event_id = Column(
        UUID(as_uuid=True),
        unique=True,
        nullable=False,
        index=True,
        default=uuid.uuid4,
    )
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    person_id = Column(Integer)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    event_type = Column(String(50), default="fall_candidate", nullable=False)
    # detection.* (mqtt_schema v2)
    class_before = Column(String(50))
    final_class = Column(String(50))
    detection_confidence = Column(Float)
    bbox_json = Column(JSONB)
    frame_width = Column(Integer)
    frame_height = Column(Integer)
    # rule.* (mqtt_schema v2)
    rule_trigger = Column(String(50))
    transition_ms = Column(Integer)
    # vlm_result.* (contract v2)
    vlm_verdict = Column(String(50))
    vlm_confidence = Column(Float)
    vlm_reason = Column(String(1000))
    image_url = Column(String(500))
    # 6 ảnh bằng chứng: [{"index": int, "offset_ms": int, "url": str}, ...]
    image_urls = Column(JSONB)
    clip_url = Column(String(500))
    # status: pending | confirmed | false_positive | merged
    status = Column(String(20), default="pending", nullable=False)
    # Ghi chú vận hành (vd: dấu vết gộp dedup multi-camera)
    note = Column(Text)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    camera = relationship("Camera", back_populates="events")
    alerts = relationship(
        "Alert", back_populates="event", cascade="all, delete-orphan"
    )
