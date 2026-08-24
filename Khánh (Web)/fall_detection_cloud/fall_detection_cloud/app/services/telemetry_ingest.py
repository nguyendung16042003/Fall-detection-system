"""Lưu telemetry từ Edge (MQTT telemetry/cam_{id}/status)."""

from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.models.telemetry import TelemetryLog
from app.schemas.telemetry import TelemetryIn


def ingest_telemetry(db: Session, data: TelemetryIn) -> TelemetryLog:
    pipeline = data.pipeline
    system = data.system
    network = data.network

    camera = db.query(Camera).filter(Camera.cam_id == data.cam_id).first()

    log = TelemetryLog(
        cam_id=data.cam_id,
        timestamp_utc=data.timestamp_utc,
        pipeline_status=pipeline.state if pipeline else None,
        fps_pgie=pipeline.fps_pgie if pipeline else None,
        fps_sgie=pipeline.fps_sgie if pipeline else None,
        active_tracks=pipeline.active_tracks if pipeline else None,
        ram_used_mb=system.ram_used_mb if system else None,
        ram_total_mb=system.ram_total_mb if system else None,
        cpu_temp_c=system.cpu_temp_c if system else None,
        gpu_temp_c=system.gpu_temp_c if system else None,
        cpu_usage_pct=system.cpu_usage_pct if system else None,
        disk_free_gb=system.disk_free_gb if system else None,
        mqtt_connected=network.mqtt_connected if network else None,
        last_event_sent_utc=(
            network.last_event_sent_utc if network else None
        ),
    )
    # cập nhật heartbeat của camera
    if camera is not None and data.timestamp_utc is not None:
        camera.last_heartbeat = data.timestamp_utc
        db.add(camera)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
