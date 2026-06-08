from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import get_db
from app.core.messaging import RoutingKey, publish
from app.models.telemetry import TelemetryLog
from app.models.user import User
from app.schemas.telemetry import (
    TelemetryCreate,
    TelemetryCreateResponse,
    TelemetryRead,
)

router = APIRouter()


@router.post(
    "",
    response_model=TelemetryCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_telemetry(
    payload: TelemetryCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> TelemetryCreateResponse:
    """Nhận telemetry từ Edge: lưu DB + publish queue telemetry_events."""
    log = TelemetryLog(
        edge_device_id=payload.edge_device_id,
        pipeline_status=payload.pipeline_status,
        fps=payload.fps,
        ram_percent=payload.ram_percent,
        cpu_usage=payload.cpu_usage,
        gpu_temp=payload.temperature_celsius,
        camera_status_json=payload.camera_status,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    publish(
        RoutingKey.TELEMETRY.value,
        {
            "edge_device_id": payload.edge_device_id,
            "timestamp": payload.timestamp,
            "pipeline_status": payload.pipeline_status,
            "fps": payload.fps,
            "ram_percent": payload.ram_percent,
            "temperature_celsius": payload.temperature_celsius,
            "camera_status": payload.camera_status,
        },
    )
    return TelemetryCreateResponse(status="received")


@router.get("/{edge_device_id}", response_model=TelemetryRead)
def get_telemetry(
    edge_device_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> TelemetryRead:
    log = (
        db.query(TelemetryLog)
        .filter(TelemetryLog.edge_device_id == edge_device_id)
        .order_by(TelemetryLog.recorded_at.desc())
        .first()
    )
    if log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không có telemetry cho thiết bị này",
        )
    return TelemetryRead(
        edge_device_id=log.edge_device_id,
        last_seen=log.recorded_at,
        pipeline_status=log.pipeline_status,
        fps=log.fps,
        ram_percent=log.ram_percent,
        temperature_celsius=log.gpu_temp,
        camera_status=log.camera_status_json,
    )
