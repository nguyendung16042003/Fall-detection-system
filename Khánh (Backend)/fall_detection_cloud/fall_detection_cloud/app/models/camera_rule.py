from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    func,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class CameraRule(Base):
    """Cấu hình Temporal State Transition Detector cho mỗi camera."""

    __tablename__ = "camera_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    time_window_sec = Column(Integer, default=2, nullable=False)
    min_lying_frames = Column(Integer, default=5, nullable=False)
    min_confidence_sgie = Column(Float, default=0.6, nullable=False)
    enable_vlm_verify = Column(Boolean, default=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    camera = relationship("Camera", back_populates="rules")

    @property
    def enabled(self) -> bool:
        """Bí danh theo API contract; ánh xạ tới cột `is_active`."""
        return self.is_active
