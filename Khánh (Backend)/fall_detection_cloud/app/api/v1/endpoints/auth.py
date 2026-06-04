from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User
from app.schemas.auth import UserLogin, Token
from app.core.security import verify_password, create_access_token

router = APIRouter()

@router.post("/login", response_model=Token)
def login(user_in: UserLogin, db: Session = Depends(get_db)):
    # 1. Truy vấn người dùng theo email từ bảng users [2]
    user = db.query(User).filter(User.email == user_in.email).first()
    
    # 2. Kiểm tra người dùng tồn tại và verify mật khẩu bằng passlib
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không chính xác",
        )
    
    # 3. Tạo JWT Token chứa user_id (UUID) [2]
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(days=7))
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
@router.post("/refresh", response_model=Token)
def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    try:
        # Giải mã token để lấy user_id (sub)
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token không hợp lệ")
    except Exception:
        raise HTTPException(status_code=401, detail="Token đã hết hạn hoặc không hợp lệ")
    
    # Cấp access token mới
    new_access_token = create_access_token(data={"sub": user_id})
    return {
        "access_token": new_access_token,
        "refresh_token": refresh_token, # Giữ nguyên refresh token cũ hoặc cấp mới tùy chính sách
        "token_type": "bearer"
    }