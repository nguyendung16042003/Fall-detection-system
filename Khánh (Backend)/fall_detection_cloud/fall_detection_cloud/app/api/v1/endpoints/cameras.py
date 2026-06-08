from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import get_db
from app.models.camera import Camera
from app.models.user import User
from app.schemas.camera import CameraCreate, CameraRead, CameraUpdate

router = APIRouter()


@router.get("", response_model=list[CameraRead])
def list_cameras(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> list[Camera]:
    return db.query(Camera).order_by(Camera.created_at.desc()).all()


@router.post(
    "", response_model=CameraRead, status_code=status.HTTP_201_CREATED
)
def create_camera(
    camera_in: CameraCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> Camera:
    camera = Camera(**camera_in.model_dump(exclude_none=True))
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def _get_camera_or_404(camera_id: UUID, db: Session) -> Camera:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy Camera",
        )
    return camera


@router.get("/{camera_id}", response_model=CameraRead)
def get_camera(
    camera_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> Camera:
    return _get_camera_or_404(camera_id, db)


@router.put("/{camera_id}", response_model=CameraRead)
def update_camera(
    camera_id: UUID,
    camera_in: CameraUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> Camera:
    camera = _get_camera_or_404(camera_id, db)
    payload = camera_in.model_dump(exclude_unset=True)
    # contract dùng `enabled`; ánh xạ sang cột `status`
    enabled = payload.pop("enabled", None)
    if enabled is not None and "status" not in payload:
        payload["status"] = "online" if enabled else "offline"
    for key, value in payload.items():
        setattr(camera, key, value)
    db.commit()
    db.refresh(camera)
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> None:
    camera = _get_camera_or_404(camera_id, db)
    db.delete(camera)
    db.commit()
