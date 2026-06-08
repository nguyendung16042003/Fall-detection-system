from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.core.database import Base


class TelemetryLog(Base):
    """Nhật ký sức khỏe thiết bị Edge (Jetson)."""

    __tablename__ = "telemetry_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    edge_device_id = Column(String(100), nullable=False, index=True)
    pipeline_status = Column(String(50))  # running / error
    fps = Column(Float)
    ram_percent = Column(Float)
    cpu_usage = Column(Float)
    gpu_temp = Column(Float)
    camera_status_json = Column(JSONB)
    recorded_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ConfidenceLog(Base):
    """Log phục vụ đánh giá mô hình và tái huấn luyện (MLOps)."""

    __tablename__ = "confidence_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(
        UUID(as_uuid=True), ForeignKey("events.id", ondelete="CASCADE")
    )
    model_version = Column(String(50))
    inference_latency_ms = Column(Float)
    environment_metadata = Column(JSONB)
    is_flagged_for_retrain = Column(Boolean, default=False, nullable=False)
