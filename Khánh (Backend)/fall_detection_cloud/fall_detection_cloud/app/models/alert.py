from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Alert(Base):
    """Thông báo cảnh báo gửi tới người chăm sóc và trạng thái xác nhận."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Tham chiếu tới events.event_id (UUID v4 của Edge) — contract v2 alert.event_id
    event_id = Column(
        UUID(as_uuid=True),
        ForeignKey("events.event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(255))
    message = Column(Text)
    fcm_sent = Column(Boolean, default=False, nullable=False)
    is_acknowledged = Column(Boolean, default=False, nullable=False, index=True)
    acknowledged_by = Column(Integer, ForeignKey("users.id"))
    acknowledged_at = Column(DateTime(timezone=True))
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    event = relationship("Event", back_populates="alerts")
