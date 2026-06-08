from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str | None = None
    role: str = "caregiver"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserBrief(BaseModel):
    """Thông tin người dùng rút gọn trả kèm khi đăng nhập (theo API contract)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str | None = Field(default=None, validation_alias="full_name")
    email: EmailStr
    role: str


class LoginResponse(Token):
    user: UserBrief


class TokenPayload(BaseModel):
    sub: str | None = None
    type: str | None = None
