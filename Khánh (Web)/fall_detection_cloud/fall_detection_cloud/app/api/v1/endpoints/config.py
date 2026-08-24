"""Cấu hình pipeline toàn cục cho Edge (contract v2 mục 6).

Tổng hợp từ bảng `camera_rules` + `settings`. Các trường không có cột DB
(dedup.window_ms, lying_ignore_after_ms) lưu runtime ở `_overrides`.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.models.camera import Camera
from app.models.camera_rule import CameraRule
from app.models.user import User
from app.schemas.config import (
    DedupCfg,
    FallDetectionCfg,
    NotificationsCfg,
    RulesConfig,
    RulesConfigUpdate,
    VLMCfg,
)

router = APIRouter()

# Override runtime cho các trường không có cột DB / không persist sau restart
_overrides: dict[str, object] = {
    "lying_ignore_after_ms": 2000,
    "dedup_window_ms": 2000,
    "vlm_enabled": settings.VLM_ENABLED,
    "fcm_enabled": settings.FCM_ENABLED,
}


def _build_config(db: Session) -> RulesConfig:
    base = db.query(CameraRule).first()
    cameras = db.query(Camera).filter(Camera.cam_id.isnot(None)).all()
    notify_map: dict[str, dict] = {}
    for cam in cameras:
        rule = (
            db.query(CameraRule)
            .filter(CameraRule.camera_id == cam.id)
            .first()
        )
        notify_map[cam.cam_id] = {
            "notify": bool(rule.is_active) if rule else True
        }

    return RulesConfig(
        fall_detection=FallDetectionCfg(
            transition_window_ms=(base.time_window_sec * 1000) if base else 2000,
            lying_ignore_after_ms=int(_overrides["lying_ignore_after_ms"]),
            min_confidence_sgie=(
                base.min_confidence_sgie if base else 0.6
            ),
            confirm_frames=base.min_lying_frames if base else 5,
        ),
        dedup=DedupCfg(window_ms=int(_overrides["dedup_window_ms"])),
        vlm=VLMCfg(
            enabled=bool(_overrides["vlm_enabled"]),
            model=settings.GEMINI_MODEL,
            timeout_s=settings.VLM_TIMEOUT_SEC,
        ),
        notifications=NotificationsCfg(
            fcm_enabled=bool(_overrides["fcm_enabled"]),
            cameras=notify_map,
        ),
    )


@router.get("/rules", response_model=RulesConfig)
def get_rules(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> RulesConfig:
    return _build_config(db)


@router.put("/rules", response_model=RulesConfig)
def update_rules(
    payload: RulesConfigUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> RulesConfig:
    fd = payload.fall_detection
    if fd is not None:
        rules = db.query(CameraRule).all()
        for rule in rules:
            if fd.transition_window_ms is not None:
                rule.time_window_sec = max(1, fd.transition_window_ms // 1000)
            if fd.confirm_frames is not None:
                rule.min_lying_frames = fd.confirm_frames
            if fd.min_confidence_sgie is not None:
                rule.min_confidence_sgie = fd.min_confidence_sgie
        if fd.lying_ignore_after_ms is not None:
            _overrides["lying_ignore_after_ms"] = fd.lying_ignore_after_ms

    if payload.dedup is not None and payload.dedup.window_ms is not None:
        _overrides["dedup_window_ms"] = payload.dedup.window_ms

    if payload.vlm is not None and payload.vlm.enabled is not None:
        _overrides["vlm_enabled"] = payload.vlm.enabled

    if payload.notifications is not None:
        notif = payload.notifications
        if notif.fcm_enabled is not None:
            _overrides["fcm_enabled"] = notif.fcm_enabled
        if notif.cameras:
            for cam_id, cfg in notif.cameras.items():
                cam = (
                    db.query(Camera)
                    .filter(Camera.cam_id == cam_id)
                    .first()
                )
                if cam is None:
                    continue
                rule = (
                    db.query(CameraRule)
                    .filter(CameraRule.camera_id == cam.id)
                    .first()
                )
                if rule is not None and "notify" in cfg:
                    rule.is_active = bool(cfg["notify"])

    db.commit()
    return _build_config(db)
