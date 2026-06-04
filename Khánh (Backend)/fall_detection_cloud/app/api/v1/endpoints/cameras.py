from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraRead, CameraUpdate

router = APIRouter()

# 2.1 Lấy danh sách camera [Mục 2.1 API Contract]
@router.get("/", response_model=List[CameraRead])
def get_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).all()

# 2.2 Tạo camera mới [Mục 2.2 API Contract]
@router.post("/", response_model=CameraRead, status_code=status.HTTP_201_CREATED)
def create_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    new_camera = Camera(**camera_in.dict())
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    return new_camera

# 2.3 Cập nhật thông tin camera [Mục 2.3 API Contract]
@router.put("/{camera_id}", response_model=CameraRead)
def update_camera(camera_id: UUID, camera_in: CameraUpdate, db: Session = Depends(get_db)):
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Không tìm thấy Camera")
    
    update_data = camera_in.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_camera, key, value)
    
    db.commit()
    db.refresh(db_camera)
    return db_camera

# 2.4 Xóa camera [Mục 2.4 API Contract]
@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: UUID, db: Session = Depends(get_db)):
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Không tìm thấy Camera")
    
    db.delete(db_camera)
    db.commit()
    return None