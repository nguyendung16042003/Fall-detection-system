from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.models.user_device import UserDevice
from app.schemas.user import DeviceRegister

router = APIRouter()


class RegisterResult(BaseModel):
    status: str = "registered"


@router.post(
    "/fcm-token",
    response_model=RegisterResult,
    status_code=status.HTTP_201_CREATED,
)
def register_fcm_token(
    payload: DeviceRegister,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> RegisterResult:
    """Đăng ký / cập nhật FCM token (contract v2 mục 8; user lấy từ JWT)."""
    device = (
        db.query(UserDevice)
        .filter(
            UserDevice.user_id == current_user.id,
            UserDevice.device_id == payload.device_id,
        )
        .first()
    )
    if device is None:
        device = UserDevice(
            user_id=current_user.id,
            device_id=payload.device_id,
            platform=payload.platform,
            fcm_token=payload.fcm_token,
        )
        db.add(device)
    else:
        device.platform = payload.platform
        device.fcm_token = payload.fcm_token
    db.commit()
    return RegisterResult()
