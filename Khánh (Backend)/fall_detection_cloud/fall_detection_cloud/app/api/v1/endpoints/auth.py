from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import (
    REFRESH_TOKEN_TYPE,
    JWTError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import (
    AccessToken,
    LoginResponse,
    RefreshRequest,
    RegisterResponse,
    UserLogin,
    UserRegister,
)

router = APIRouter()

_ACCESS_TTL_SEC = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(user_in: UserRegister, db: Session = Depends(get_db)) -> User:
    existing = (
        db.query(User)
        .filter(
            (User.email == user_in.email)
            | (User.username == user_in.username)
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username hoặc email đã được đăng ký",
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        role=user_in.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=LoginResponse)
def login(user_in: UserLogin, db: Session = Depends(get_db)) -> LoginResponse:
    user = (
        db.query(User).filter(User.username == user_in.username).first()
    )
    if not user or not verify_password(
        user_in.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không chính xác",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Tài khoản bị khóa"
        )

    subject = str(user.id)
    return LoginResponse(
        access_token=create_access_token(subject),
        refresh_token=create_refresh_token(subject),
        expires_in=_ACCESS_TTL_SEC,
    )


@router.post("/refresh", response_model=AccessToken)
def refresh(
    payload: RefreshRequest, db: Session = Depends(get_db)
) -> AccessToken:
    try:
        data = decode_token(payload.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token hết hạn hoặc không hợp lệ",
        ) from None

    if data.get("type") != REFRESH_TOKEN_TYPE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không phải refresh token",
        )

    subject = data.get("sub")
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token không hợp lệ",
        )

    return AccessToken(
        access_token=create_access_token(subject),
        expires_in=_ACCESS_TTL_SEC,
    )
