from datetime import datetime

from pydantic import BaseModel, Field


class TelemetryCreate(BaseModel):
    """Payload telemetry từ Edge (api_contract.md mục 6.1)."""

    schema_version: str | None = None
    edge_device_id: str = Field(..., examples=["jetson_nano_01"])
    timestamp: datetime | None = None
    pipeline_status: str | None = Field(default=None, examples=["running"])
    fps: float | None = None
    ram_percent: float | None = None
    cpu_usage: float | None = None
    temperature_celsius: float | None = Field(default=None, examples=[68])
    camera_status: dict[str, str] | None = None


class TelemetryCreateResponse(BaseModel):
    status: str = "received"


class TelemetryRead(BaseModel):
    edge_device_id: str
    last_seen: datetime
    pipeline_status: str | None = None
    fps: float | None = None
    ram_percent: float | None = None
    temperature_celsius: float | None = None
    camera_status: dict[str, str] | None = None
