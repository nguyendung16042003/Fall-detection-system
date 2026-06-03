from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.camera_rule import CameraRule
from app.schemas.camera_rule import CameraRuleRead, CameraRuleUpdate
from uuid import UUID

router = APIRouter()

@router.get("/{camera_id}", response_model=CameraRuleRead)
def get_camera_rule(camera_id: UUID, db: Session = Depends(get_db)):
    rule = db.query(CameraRule).filter(CameraRule.camera_id == camera_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Không tìm thấy quy tắc cho camera này")
    return rule

@router.put("/{camera_id}", response_model=CameraRuleRead)
def update_camera_rule(camera_id: UUID, rule_in: CameraRuleUpdate, db: Session = Depends(get_db)):
    rule = db.query(CameraRule).filter(CameraRule.camera_id == camera_id).first()
    if not rule:
        # Nếu chưa có thì tạo mới cấu hình mặc định
        rule = CameraRule(camera_id=camera_id, **rule_in.dict())
        db.add(rule)
    else:
        # Cập nhật các thông số mới
        for key, value in rule_in.dict().items():
            setattr(rule, key, value)
    
    db.commit()
    db.refresh(rule)
    return rule