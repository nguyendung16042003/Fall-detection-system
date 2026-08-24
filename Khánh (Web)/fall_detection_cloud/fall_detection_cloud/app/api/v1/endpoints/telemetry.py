from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.models.telemetry import TelemetryLog
from app.models.user import User
from app.schemas.telemetry import TelemetryLatest

router = APIRouter()


@router.get("/{cam_id}/latest", response_model=TelemetryLatest)
def get_latest_telemetry(
    cam_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> TelemetryLatest:
    """Telemetry mới nhất của camera (contract v2 mục 5).

    Telemetry được nạp qua MQTT (telemetry/cam_{id}/status); endpoint này chỉ
    đọc cho Duy/Mobile.
    """
    log = (
        db.query(TelemetryLog)
        .filter(TelemetryLog.cam_id == cam_id)
        .order_by(TelemetryLog.recorded_at.desc())
        .first()
    )
    if log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chưa có telemetry cho camera này",
        )

    ts = log.timestamp_utc or log.recorded_at
    is_online = False
    if ts is not None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        window = timedelta(seconds=settings.TELEMETRY_ONLINE_WINDOW_SEC)
        is_online = (datetime.now(timezone.utc) - ts) <= window

    return TelemetryLatest(
        cam_id=cam_id,
        timestamp_utc=log.timestamp_utc,
        is_online=is_online,
        pipeline_state=log.pipeline_status,
        fps_pgie=log.fps_pgie,
        fps_sgie=log.fps_sgie,
        active_tracks=log.active_tracks,
        ram_used_mb=log.ram_used_mb,
        ram_total_mb=log.ram_total_mb,
        cpu_temp_c=log.cpu_temp_c,
        gpu_temp_c=log.gpu_temp_c,
    )


@router.get("/{cam_id}")
def list_telemetry_logs(
    cam_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> list[dict]:
    """Lấy danh sách telemetry logs của camera.

    Phân trang với limit và offset.
    """
    logs = (
        db.query(TelemetryLog)
        .filter(TelemetryLog.cam_id == cam_id)
        .order_by(TelemetryLog.recorded_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    
    return [
        {
            "id": log.id,
            "cam_id": log.cam_id,
            "timestamp_utc": log.timestamp_utc,
            "recorded_at": log.recorded_at,
            "pipeline_status": log.pipeline_status,
            "fps_pgie": log.fps_pgie,
            "fps_sgie": log.fps_sgie,
            "active_tracks": log.active_tracks,
            "ram_used_mb": log.ram_used_mb,
            "ram_total_mb": log.ram_total_mb,
            "cpu_temp_c": log.cpu_temp_c,
            "gpu_temp_c": log.gpu_temp_c,
            "cpu_usage_pct": log.cpu_usage_pct,
            "disk_free_gb": log.disk_free_gb,
            "mqtt_connected": log.mqtt_connected,
            "last_event_sent_utc": log.last_event_sent_utc,
        }
        for log in logs
    ]
