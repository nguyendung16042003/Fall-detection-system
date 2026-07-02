from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import get_db
from app.models.camera import Camera
from app.models.camera_rule import CameraRule
from app.models.user import User
from app.schemas.camera_rule import CameraRuleRead, CameraRuleUpdate

router = APIRouter()


@router.get("/cameras")
def list_all_rules(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> list[dict]:
    """Lấy danh sách tất cả rules của tất cả cameras."""
    rules = (
        db.query(CameraRule)
        .join(Camera)
        .all()
    )
    
    return [
        {
            "id": rule.id,
            "camera_id": rule.camera_id,
            "cam_id": rule.camera.cam_id,
            "camera_name": rule.camera.name,
            "time_window_sec": rule.time_window_sec,
            "min_lying_frames": rule.min_lying_frames,
            "high_confidence_threshold": rule.high_confidence_threshold,
            "low_confidence_threshold": rule.low_confidence_threshold,
            "enable_vlm_verify": rule.enable_vlm_verify,
            "is_active": rule.is_active,
            "updated_at": rule.updated_at,
        }
        for rule in rules
    ]


@router.get("/cameras/{cam_id}", response_model=CameraRuleRead)
def get_camera_rule_by_cam_id(
    cam_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> CameraRule:
    """Lấy rule theo cam_id string (cho Edge)."""
    camera = db.query(Camera).filter(Camera.cam_id == cam_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy Camera",
        )

    rule = (
        db.query(CameraRule)
        .filter(CameraRule.camera_id == camera.id)
        .first()
    )
    if rule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy quy tắc cho camera này",
        )
    return rule


@router.put("/cameras/{cam_id}", response_model=CameraRuleRead)
def upsert_camera_rule_by_cam_id(
    cam_id: str,
    rule_in: CameraRuleUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> CameraRule:
    """Cập nhật hoặc tạo rule theo cam_id string (cho Edge/App)."""
    camera = db.query(Camera).filter(Camera.cam_id == cam_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy Camera",
        )

    rule = (
        db.query(CameraRule)
        .filter(CameraRule.camera_id == camera.id)
        .first()
    )
    payload = rule_in.model_dump(exclude_unset=True, exclude_none=True)
    # contract dùng `enabled`, DB lưu ở cột `is_active`
    if "enabled" in payload:
        payload["is_active"] = payload.pop("enabled")

    if rule is None:
        rule = CameraRule(camera_id=camera.id, **payload)
        db.add(rule)
    else:
        for key, value in payload.items():
            setattr(rule, key, value)

    db.commit()
    db.refresh(rule)
    return rule


@router.get("/{camera_id}", response_model=CameraRuleRead)
def get_camera_rule(
    camera_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> CameraRule:
    """Lấy rule theo camera_id UUID (legacy endpoint)."""
    rule = (
        db.query(CameraRule)
        .filter(CameraRule.camera_id == camera_id)
        .first()
    )
    if rule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy quy tắc cho camera này",
        )
    return rule


@router.put("/{camera_id}", response_model=CameraRuleRead)
def upsert_camera_rule(
    camera_id: UUID,
    rule_in: CameraRuleUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> CameraRule:
    """Cập nhật hoặc tạo rule theo camera_id UUID (legacy endpoint)."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy Camera",
        )

    rule = (
        db.query(CameraRule)
        .filter(CameraRule.camera_id == camera_id)
        .first()
    )
    payload = rule_in.model_dump(exclude_unset=True, exclude_none=True)
    # contract dùng `enabled`, DB lưu ở cột `is_active`
    if "enabled" in payload:
        payload["is_active"] = payload.pop("enabled")

    if rule is None:
        rule = CameraRule(camera_id=camera_id, **payload)
        db.add(rule)
    else:
        for key, value in payload.items():
            setattr(rule, key, value)

    db.commit()
    db.refresh(rule)
    return rule
