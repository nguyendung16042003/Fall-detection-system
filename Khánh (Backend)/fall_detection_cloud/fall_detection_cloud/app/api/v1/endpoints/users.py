from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.models.user_device import UserDevice
from app.schemas.user import (
    DeviceRead,
    DeviceRegister,
    UserRead,
    UserUpdate,
)

router = APIRouter()


@router.get("/me", response_model=UserRead)
def read_me(current_user: User = Depends(get_current_active_user)) -> User:
    return current_user


@router.put("/me", response_model=UserRead)
def update_me(
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> User:
    for key, value in user_in.model_dump(exclude_unset=True).items():
        setattr(current_user, key, value)
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get("/me/devices", response_model=list[DeviceRead])
def list_my_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> list[UserDevice]:
    return (
        db.query(UserDevice)
        .filter(UserDevice.user_id == current_user.id)
        .all()
    )


@router.put("/me/fcm-token", response_model=DeviceRead)
def upsert_fcm_token(
    device_in: DeviceRegister,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> UserDevice:
    """Đăng ký hoặc cập nhật FCM token cho thiết bị của người dùng.

    Định danh theo (user_id, device_id) để hỗ trợ đa thiết bị.
    """
    device = (
        db.query(UserDevice)
        .filter(
            UserDevice.user_id == current_user.id,
            UserDevice.device_id == device_in.device_id,
        )
        .first()
    )
    if device is None:
        device = UserDevice(
            user_id=current_user.id,
            device_id=device_in.device_id,
            fcm_token=device_in.fcm_token,
            platform=device_in.platform,
        )
        db.add(device)
    else:
        device.fcm_token = device_in.fcm_token
        device.platform = device_in.platform

    db.commit()
    db.refresh(device)
    return device


@router.delete(
    "/me/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_my_device(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> None:
    db.query(UserDevice).filter(
        UserDevice.user_id == current_user.id,
        UserDevice.device_id == device_id,
    ).delete()
    db.commit()
