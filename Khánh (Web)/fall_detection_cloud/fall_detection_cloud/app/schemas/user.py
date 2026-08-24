from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime


class UserUpdate(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    role: str | None = None
    is_active: bool | None = None


class DeviceRegister(BaseModel):
    """Đăng ký / cập nhật FCM token cho một thiết bị của người dùng."""

    device_id: str = Field(..., examples=["android_device_001"])
    fcm_token: str = Field(..., examples=["fcm_token_abc..."])
    platform: str | None = Field(default=None, examples=["android", "ios"])


class DeviceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    device_id: str
    platform: str | None = None
    fcm_token: str
    updated_at: datetime
