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


@router.get("/{camera_id}", response_model=CameraRuleRead)
def get_camera_rule(
    camera_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> CameraRule:
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
