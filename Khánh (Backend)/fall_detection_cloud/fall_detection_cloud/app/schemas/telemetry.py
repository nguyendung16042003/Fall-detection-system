from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PipelineIn(BaseModel):
    state: str | None = None
    fps_pgie: float | None = None
    fps_sgie: float | None = None
    active_tracks: int | None = None


class SystemIn(BaseModel):
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
    cpu_usage_pct: float | None = None
    disk_free_gb: float | None = None


class NetworkIn(BaseModel):
    mqtt_connected: bool | None = None
    last_event_sent_utc: datetime | None = None


class TelemetryIn(BaseModel):
    """telemetry/cam_{id}/status (mqtt_schema v2)."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str | None = None
    cam_id: str = Field(..., examples=["cam_01"])
    timestamp_utc: datetime | None = None
    pipeline: PipelineIn | None = None
    system: SystemIn | None = None
    network: NetworkIn | None = None


class TelemetryLatest(BaseModel):
    """Response GET /api/telemetry/{cam_id}/latest (contract v2 mục 5)."""

    cam_id: str
    timestamp_utc: datetime | None = None
    is_online: bool
    pipeline_state: str | None = None
    fps_pgie: float | None = None
    fps_sgie: float | None = None
    active_tracks: int | None = None
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
