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
    if (
        db.query(Camera).filter(Camera.cam_id == camera_in.cam_id).first()
        is not None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="cam_id đã tồn tại",
        )
    camera = Camera(
        cam_id=camera_in.cam_id,
        name=camera_in.name,
        rtsp_url=camera_in.rtsp_url,
        location=camera_in.location,
        edge_device_id=camera_in.edge_device_id or "",
        is_active=camera_in.is_active,
        status="online" if camera_in.is_active else "offline",
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def _get_camera_or_404(cam_id: str, db: Session) -> Camera:
    camera = db.query(Camera).filter(Camera.cam_id == cam_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy Camera",
        )
    return camera


@router.get("/{cam_id}", response_model=CameraRead)
def get_camera(
    cam_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> Camera:
    return _get_camera_or_404(cam_id, db)


@router.put("/{cam_id}", response_model=CameraRead)
def update_camera(
    cam_id: str,
    camera_in: CameraUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> Camera:
    camera = _get_camera_or_404(cam_id, db)
    payload = camera_in.model_dump(exclude_unset=True)
    if "is_active" in payload:
        camera.status = "online" if payload["is_active"] else "offline"
    for key, value in payload.items():
        setattr(camera, key, value)
    db.commit()
    db.refresh(camera)
    return camera


@router.delete("/{cam_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    cam_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> None:
    camera = _get_camera_or_404(cam_id, db)
    db.delete(camera)
    db.commit()
