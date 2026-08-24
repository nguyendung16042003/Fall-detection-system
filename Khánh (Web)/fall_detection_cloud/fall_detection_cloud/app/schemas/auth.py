from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    """Đăng ký tài khoản (contract v2 mục 1)."""

    username: str = Field(..., min_length=3, max_length=150)
    password: str = Field(..., min_length=6, max_length=128)
    email: EmailStr
    role: str = "caregiver"


class UserLogin(BaseModel):
    """Đăng nhập (contract v2: username + password)."""

    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessToken(BaseModel):
    """Response của /auth/refresh (contract v2)."""

    access_token: str
    expires_in: int


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(Token):
    """Response của /auth/login (contract v2)."""

    expires_in: int


class RegisterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    created_at: datetime


class TokenPayload(BaseModel):
    sub: str | None = None
    type: str | None = None
