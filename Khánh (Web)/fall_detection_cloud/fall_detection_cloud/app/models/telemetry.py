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
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import Base


class TelemetryLog(Base):
    """Nhật ký sức khỏe thiết bị Edge (Jetson) — mqtt_schema v2."""

    __tablename__ = "telemetry_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    # Contract v2: telemetry theo từng cam_id (MQTT telemetry/cam_{id}/status)
    cam_id = Column(String(50), index=True, nullable=False)
    timestamp_utc = Column(DateTime(timezone=True))

    # pipeline.*
    pipeline_status = Column(String(50))  # running / stopped / error / starting
    fps_pgie = Column(Float)
    fps_sgie = Column(Float)
    active_tracks = Column(Integer)

    # system.*
    ram_used_mb = Column(Float)
    ram_total_mb = Column(Float)
    cpu_temp_c = Column(Float)
    gpu_temp_c = Column(Float)
    cpu_usage_pct = Column(Float)
    disk_free_gb = Column(Float)

    # network.*
    mqtt_connected = Column(Boolean)
    last_event_sent_utc = Column(DateTime(timezone=True))

    recorded_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ConfidenceLog(Base):
    """Log phục vụ đánh giá mô hình và tái huấn luyện (MLOps)."""

    __tablename__ = "confidence_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey("events.id", ondelete="CASCADE")
    )
    model_version = Column(String(50))
    inference_latency_ms = Column(Float)
    environment_metadata = Column(JSONB)
    is_flagged_for_retrain = Column(Boolean, default=False, nullable=False)
